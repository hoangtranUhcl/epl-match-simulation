import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import psycopg2
from sqlalchemy import create_engine

# Database connection parameters (replace with your own)
db_params = {
    'dbname': 'EPLDatabase',
    'user': 'postgres',
    'password': '2025',
    'host': 'localhost',
    'port': '5432'
}

# Create connection string for SQLAlchemy
conn_string = f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"
engine = create_engine(conn_string)

# 1. Load data from PostgreSQL
queries = {
    'match_results': "SELECT * FROM match_results",
    'player_values': "SELECT * FROM player_values",
    'teams': "SELECT * FROM teams",
    'fixtures': "SELECT * FROM fixtures WHERE fixture_date >= '2025-02-25'"
}

match_results = pd.read_sql(queries['match_results'], engine)
player_values = pd.read_sql(queries['player_values'], engine)
teams = pd.read_sql(queries['teams'], engine)
fixtures = pd.read_sql(queries['fixtures'], engine)

# 2. Data Preparation
match_results['date'] = pd.to_datetime(match_results['date'])
player_values['season'] = player_values['season'].astype(int)
fixtures['fixture_date'] = pd.to_datetime(fixtures['fixture_date'])

# Filter match results up to Feb 25, 2025
current_date = pd.to_datetime('2025-02-25')
match_results = match_results[match_results['date'] <= current_date]

# Merge team names
match_results = match_results.merge(teams[['team_id', 'team_name']], 
                                    left_on='home_team_id', right_on='team_id', 
                                    how='left').rename(columns={'team_name': 'home_team_name'}).drop('team_id', axis=1)
match_results = match_results.merge(teams[['team_id', 'team_name']], 
                                    left_on='away_team_id', right_on='team_id', 
                                    how='left').rename(columns={'team_name': 'away_team_name'}).drop('team_id', axis=1)

# 3. Feature Engineering
def team_performance(df, team_id, date):
    team_matches = df[((df['home_team_id'] == team_id) | (df['away_team_id'] == team_id)) & 
                      (df['date'] < date)]
    if len(team_matches) == 0:
        return pd.Series({'avg_goals_scored': 0, 'avg_goals_conceded': 0, 'win_rate': 0})
    
    home_matches = team_matches[team_matches['home_team_id'] == team_id]
    away_matches = team_matches[team_matches['away_team_id'] == team_id]
    
    goals_scored = (home_matches['home_goals'].sum() + away_matches['away_goals'].sum()) / len(team_matches)
    goals_conceded = (home_matches['away_goals'].sum() + away_matches['home_goals'].sum()) / len(team_matches)
    wins = len(home_matches[home_matches['result'] == 1]) + len(away_matches[away_matches['result'] == -1])
    win_rate = wins / len(team_matches) if len(team_matches) > 0 else 0
    
    return pd.Series({'avg_goals_scored': goals_scored, 'avg_goals_conceded': goals_conceded, 'win_rate': win_rate})

player_values_agg = player_values.groupby(['team_id', 'season']).agg({'market_value': 'mean'}).reset_index()
player_values_agg = player_values_agg.rename(columns={'market_value': 'avg_player_value'})

match_features = []
for _, row in match_results.iterrows():
    home_stats = team_performance(match_results, row['home_team_id'], row['date'])
    away_stats = team_performance(match_results, row['away_team_id'], row['date'])
    
    season = row['date'].year - 1 if row['date'].month < 8 else row['date'].year
    home_value = player_values_agg[(player_values_agg['team_id'] == row['home_team_id']) & 
                                   (player_values_agg['season'] <= season)]['avg_player_value'].max()
    away_value = player_values_agg[(player_values_agg['team_id'] == row['away_team_id']) & 
                                   (player_values_agg['season'] <= season)]['avg_player_value'].max()
    
    match_features.append({
        'match_id': row['match_id'],
        'home_team_id': row['home_team_id'],
        'away_team_id': row['away_team_id'],
        'home_avg_goals_scored': home_stats['avg_goals_scored'],
        'home_avg_goals_conceded': home_stats['avg_goals_conceded'],
        'home_win_rate': home_stats['win_rate'],
        'away_avg_goals_scored': away_stats['avg_goals_scored'],
        'away_avg_goals_conceded': away_stats['avg_goals_conceded'],
        'away_win_rate': away_stats['win_rate'],
        'home_avg_player_value': home_value if not pd.isna(home_value) else 0,
        'away_avg_player_value': away_value if not pd.isna(away_value) else 0,
        'result': row['result']
    })

match_results_features = pd.DataFrame(match_features)

# 4. Prepare target variable
le = LabelEncoder()
match_results_features['result'] = le.fit_transform(match_results_features['result'] + 1)  # 0=away, 1=draw, 2=home

X = match_results_features.drop(['match_id', 'home_team_id', 'away_team_id', 'result'], axis=1)
y = match_results_features['result']

# 5. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 6. Train the Model with Logistic Regression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# 7. Evaluate Model Accuracy
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy with Logistic Regression: {accuracy * 100:.2f}%")

# Detailed classification report
report = classification_report(y_test, y_pred, target_names=['Away Win', 'Draw', 'Home Win'])
print("\nDetailed Classification Report:")
print(report)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(pd.DataFrame(cm, index=['Away Win', 'Draw', 'Home Win'], columns=['Pred Away', 'Pred Draw', 'Pred Home']))

# 8. Predict Upcoming Matches
fixture_features = []
for _, row in fixtures.iterrows():
    home_stats = team_performance(match_results, row['home_team_id'], row['fixture_date'])
    away_stats = team_performance(match_results, row['away_team_id'], row['fixture_date'])
    
    season = row['fixture_date'].year - 1 if row['fixture_date'].month < 8 else row['fixture_date'].year
    home_value = player_values_agg[(player_values_agg['team_id'] == row['home_team_id']) & 
                                   (player_values_agg['season'] <= season)]['avg_player_value'].max()
    away_value = player_values_agg[(player_values_agg['team_id'] == row['away_team_id']) & 
                                   (player_values_agg['season'] <= season)]['avg_player_value'].max()
    
    fixture_features.append({
        'home_team_id': row['home_team_id'],
        'away_team_id': row['away_team_id'],
        'home_avg_goals_scored': home_stats['avg_goals_scored'],
        'home_avg_goals_conceded': home_stats['avg_goals_conceded'],
        'home_win_rate': home_stats['win_rate'],
        'away_avg_goals_scored': away_stats['avg_goals_scored'],
        'away_avg_goals_conceded': away_stats['avg_goals_conceded'],
        'away_win_rate': away_stats['win_rate'],
        'home_avg_player_value': home_value if not pd.isna(home_value) else 0,
        'away_avg_player_value': away_value if not pd.isna(away_value) else 0
    })

fixtures_features = pd.DataFrame(fixture_features)
X_fixtures = fixtures_features.drop(['home_team_id', 'away_team_id'], axis=1)
X_fixtures_scaled = scaler.transform(X_fixtures)

fixture_predictions = model.predict(X_fixtures_scaled)
fixture_predictions_proba = model.predict_proba(X_fixtures_scaled)

# Create results DataFrame with all probabilities
results = fixtures[['fixture_date', 'home_team_id', 'away_team_id']].copy()
results['predicted_result'] = le.inverse_transform(fixture_predictions) - 1
results['home_win_prob'] = fixture_predictions_proba[:, 2] * 100  # Home win (originally encoded as 2)
results['draw_prob'] = fixture_predictions_proba[:, 1] * 100      # Draw (originally encoded as 1)
results['away_win_prob'] = fixture_predictions_proba[:, 0] * 100  # Away win (originally encoded as 0)
results['winner'] = results['predicted_result'].apply(lambda x: 'Home' if x == 1 else ('Away' if x == -1 else 'Draw'))

# Merge team names
results = results.merge(teams[['team_id', 'team_name']], left_on='home_team_id', right_on='team_id', how='left')\
                 .rename(columns={'team_name': 'home_team_name'}).drop('team_id', axis=1)
results = results.merge(teams[['team_id', 'team_name']], left_on='away_team_id', right_on='team_id', how='left')\
                 .rename(columns={'team_name': 'away_team_name'}).drop('team_id', axis=1)

# Select final columns
results = results[['fixture_date', 'home_team_name', 'away_team_name', 'winner', 
                  'home_win_prob', 'draw_prob', 'away_win_prob']]

# Round probabilities to 2 decimal places
results[['home_win_prob', 'draw_prob', 'away_win_prob']] = results[['home_win_prob', 'draw_prob', 'away_win_prob']].round(2)

# 9. Display Sample Predictions
print("\nSample Upcoming Match Predictions:")
print(results.head(5))

# 10. Save to PostgreSQL
create_table_query = """
CREATE TABLE IF NOT EXISTS match_predictions (
    fixture_date DATE,
    home_team_name VARCHAR(50),
    away_team_name VARCHAR(50),
    winner VARCHAR(10),
    home_win_prob FLOAT,
    draw_prob FLOAT,
    away_win_prob FLOAT
);
"""
with psycopg2.connect(**db_params) as conn:
    with conn.cursor() as cur:
        cur.execute(create_table_query)
    conn.commit()

results.to_sql('match_predictions', engine, if_exists='replace', index=False)
print("\nPredictions saved to 'match_predictions' table in PostgreSQL.")