import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import psycopg2
from sqlalchemy import create_engine, text
from typing import Dict, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "host": "68.74.165.203",
    "port": "5432",
    "database": "EPL",
    "user": "postgres",
    "password": "7410"
}

def get_db_connection():
    """Create and return a database connection."""
    conn_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    return create_engine(conn_string)

def load_data(engine) -> Dict[str, pd.DataFrame]:
    """Load all required data from the database."""
    queries = {
        'match_results': "SELECT * FROM match_results",
        'player_values': "SELECT * FROM player_values",
        'teams': "SELECT * FROM teams",
        'fixtures': """
            SELECT fixture_date, match_id, time, home_team_id, away_team_id 
            FROM matches_2025 
            WHERE fixture_date >= '2025-02-25' 
            ORDER BY fixture_date, time
        """
    }
    return {key: pd.read_sql(query, engine) for key, query in queries.items()}

def prepare_data(match_results: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Prepare and clean the match results data."""
    match_results['date'] = pd.to_datetime(match_results['date'])
    current_date = pd.to_datetime('2025-02-25')
    match_results = match_results[match_results['date'] <= current_date]
    
    # Merge team names in one operation
    match_results = match_results.merge(
        teams[['team_id', 'team_name']], 
        left_on='home_team_id', 
        right_on='team_id', 
        how='left'
    ).rename(columns={'team_name': 'home_team_name'}).drop('team_id', axis=1)
    
    match_results = match_results.merge(
        teams[['team_id', 'team_name']], 
        left_on='away_team_id', 
        right_on='team_id', 
        how='left'
    ).rename(columns={'team_name': 'away_team_name'}).drop('team_id', axis=1)
    
    return match_results

def calculate_team_performance(df: pd.DataFrame, team_id: int, date: pd.Timestamp, window: int = 5) -> pd.Series:
    """Calculate team performance metrics for a given window."""
    team_matches = df[
        ((df['home_team_id'] == team_id) | (df['away_team_id'] == team_id)) & 
        (df['date'] < date)
    ].sort_values('date', ascending=False).head(window)
    
    if len(team_matches) == 0:
        return pd.Series({'avg_goals_scored': 0, 'avg_goals_conceded': 0, 'win_rate': 0, 'draw_rate': 0})
    
    home_matches = team_matches[team_matches['home_team_id'] == team_id]
    away_matches = team_matches[team_matches['away_team_id'] == team_id]
    
    total_matches = len(team_matches)
    goals_scored = (home_matches['home_goals'].sum() + away_matches['away_goals'].sum()) / total_matches
    goals_conceded = (home_matches['away_goals'].sum() + away_matches['home_goals'].sum()) / total_matches
    wins = len(home_matches[home_matches['result'] == 1]) + len(away_matches[away_matches['result'] == -1])
    draws = len(team_matches[team_matches['result'] == 0])
    
    return pd.Series({
        'avg_goals_scored': goals_scored,
        'avg_goals_conceded': goals_conceded,
        'win_rate': wins / total_matches,
        'draw_rate': draws / total_matches
    })

def create_features(match_results: pd.DataFrame, player_values: pd.DataFrame, historical_data: pd.DataFrame = None, is_fixture: bool = False) -> pd.DataFrame:
    """Create features for the model."""
    # Ensure season is integer type
    player_values['season'] = player_values['season'].astype(int)
    player_values_agg = player_values.groupby(['team_id', 'season'])['market_value'].mean().reset_index()
    player_values_agg = player_values_agg.rename(columns={'market_value': 'avg_player_value'})
    
    # Use historical data if provided, otherwise use match_results
    historical_data = historical_data if historical_data is not None else match_results
    
    features = []
    for _, row in match_results.iterrows():
        # Handle date differently for fixtures vs match results
        date = row['fixture_date'] if is_fixture else row['date']
        
        # Use historical data for performance calculation
        home_stats = calculate_team_performance(historical_data, row['home_team_id'], date)
        away_stats = calculate_team_performance(historical_data, row['away_team_id'], date)
        
        season = date.year - 1 if date.month < 8 else date.year
        home_value = player_values_agg[
            (player_values_agg['team_id'] == row['home_team_id']) & 
            (player_values_agg['season'] <= season)
        ]['avg_player_value'].max()
        
        away_value = player_values_agg[
            (player_values_agg['team_id'] == row['away_team_id']) & 
            (player_values_agg['season'] <= season)
        ]['avg_player_value'].max()
        
        feature_dict = {
            'match_id': row['match_id'],
            'home_team_id': row['home_team_id'],
            'away_team_id': row['away_team_id'],
            'home_avg_goals_scored': home_stats['avg_goals_scored'],
            'home_avg_goals_conceded': home_stats['avg_goals_conceded'],
            'home_win_rate': home_stats['win_rate'],
            'home_draw_rate': home_stats['draw_rate'],
            'away_avg_goals_scored': away_stats['avg_goals_scored'],
            'away_avg_goals_conceded': away_stats['avg_goals_conceded'],
            'away_win_rate': away_stats['win_rate'],
            'away_draw_rate': away_stats['draw_rate'],
            'home_avg_player_value': home_value if not pd.isna(home_value) else 0,
            'away_avg_player_value': away_value if not pd.isna(away_value) else 0
        }
        
        if not is_fixture:
            feature_dict['result'] = row['result']
            
        features.append(feature_dict)
    
    return pd.DataFrame(features)

def train_model(X: pd.DataFrame, y: pd.Series) -> Tuple[XGBClassifier, StandardScaler]:
    """Train the XGBoost model with optimized parameters."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    
    # Split data directly
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train model with adjusted parameters
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        n_estimators=150,  # Increased number of trees
        learning_rate=0.05,  # Lower learning rate
        max_depth=7,  # Increased depth
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,  # Removed class weights
        gamma=0.5  # Added regularization
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    return model, scaler

def save_predictions(results: pd.DataFrame, db_params: Dict[str, str]):
    """Save predictions to the database efficiently."""
    with psycopg2.connect(**db_params) as conn:
        with conn.cursor() as cur:
            # Create temporary table
            cur.execute("""
                CREATE TEMP TABLE temp_match_prediction (
                    match_id INTEGER,
                    fixture_date TIMESTAMP,
                    time TIME,
                    home_team_name VARCHAR(100),
                    away_team_name VARCHAR(100),
                    winner VARCHAR(10),
                    home_win_prob NUMERIC,
                    draw_prob NUMERIC,
                    away_win_prob NUMERIC
                );
            """)
            
            # Bulk insert predictions
            for _, row in results.iterrows():
                cur.execute("""
                    INSERT INTO temp_match_prediction 
                    (match_id, fixture_date, time, home_team_name, away_team_name, winner, home_win_prob, draw_prob, away_win_prob)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
                """, (
                    row['match_id'],
                    row['fixture_date'],
                    row['time'],
                    row['home_team_name'],
                    row['away_team_name'],
                    row['winner'],
                    row['home_win_prob'],
                    row['draw_prob'],
                    row['away_win_prob']
                ))
            
            # Update predictions in one transaction
            cur.execute("""
                UPDATE match_predictions
                SET 
                    fixture_date = temp_match_prediction.fixture_date,
                    time = temp_match_prediction.time,
                    home_team_name = temp_match_prediction.home_team_name,
                    away_team_name = temp_match_prediction.away_team_name,
                    winner = temp_match_prediction.winner,
                    home_win_prob = temp_match_prediction.home_win_prob,
                    draw_prob = temp_match_prediction.draw_prob,
                    away_win_prob = temp_match_prediction.away_win_prob
                FROM temp_match_prediction
                WHERE match_predictions.match_id = temp_match_prediction.match_id;
            """)
            
            conn.commit()
            logger.info("Predictions updated in 'match_predictions' table")

def main():
    """Main execution function."""
    try:
        # Initialize database connection
        engine = get_db_connection()
        
        # Load data
        data = load_data(engine)
        match_results = prepare_data(data['match_results'], data['teams'])
        
        # Create features
        features = create_features(match_results, data['player_values'])
        
        # Prepare target variable
        le = LabelEncoder()
        features['result'] = le.fit_transform(features['result'] + 1)
        
        # Train model
        X = features.drop(['match_id', 'home_team_id', 'away_team_id', 'result'], axis=1)
        y = features['result']
        model, scaler = train_model(X, y)
        
        # Generate predictions for fixtures
        fixture_features = create_features(
            data['fixtures'], 
            data['player_values'], 
            historical_data=match_results,
            is_fixture=True
        )
        X_fixtures = fixture_features.drop(['match_id', 'home_team_id', 'away_team_id'], axis=1)
        X_fixtures_scaled = scaler.transform(X_fixtures)
        
        predictions = model.predict(X_fixtures_scaled)
        predictions_proba = model.predict_proba(X_fixtures_scaled)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'fixture_date': data['fixtures']['fixture_date'],  # Keep original datetime format
            'time': data['fixtures']['time'],
            'match_id': data['fixtures']['match_id'],
            'home_team_id': data['fixtures']['home_team_id'],
            'away_team_id': data['fixtures']['away_team_id'],
            'predicted_result': le.inverse_transform(predictions) - 1,
            'home_win_prob': predictions_proba[:, 2] * 100,
            'draw_prob': predictions_proba[:, 1] * 100,
            'away_win_prob': predictions_proba[:, 0] * 100
        })
        
        # Add team names and winner
        results = results.merge(
            data['teams'][['team_id', 'team_name']], 
            left_on='home_team_id', 
            right_on='team_id', 
            how='left'
        ).rename(columns={'team_name': 'home_team_name'}).drop('team_id', axis=1)
        
        results = results.merge(
            data['teams'][['team_id', 'team_name']], 
            left_on='away_team_id', 
            right_on='team_id', 
            how='left'
        ).rename(columns={'team_name': 'away_team_name'}).drop('team_id', axis=1)
        
        results['winner'] = results['predicted_result'].apply(
            lambda x: 'Home' if x == 1 else ('Away' if x == -1 else 'Draw')
        )
        
        # Select and sort results
        results = results[[
            'fixture_date', 'time', 'match_id', 'home_team_name', 
            'away_team_name', 'winner', 'home_win_prob', 
            'draw_prob', 'away_win_prob'
        ]].sort_values(['fixture_date', 'time'])
        
        # Round probabilities
        results[['home_win_prob', 'draw_prob', 'away_win_prob']] = results[
            ['home_win_prob', 'draw_prob', 'away_win_prob']
        ].round(2)
        
        # Save predictions
        save_predictions(results, DB_CONFIG)
        
        logger.info("Model execution completed successfully")
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()