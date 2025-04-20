results = fixtures[['fixture_date', 'match_id', 'home_team_id', 'away_team_id', 'time']].copy()
results['predicted_result'] = le.inverse_transform(fixture_predictions) - 1
results['home_win_prob'] = fixture_predictions_proba[:, 2] * 100  # Home win (originally encoded as 2)
results['draw_prob'] = fixture_predictions_proba[:, 1] * 100      # Draw (originally encoded as 1)
results['away_win_prob'] = fixture_predictions_proba[:, 0] * 100  # Away win (originally encoded as 0)
results['winner'] = results['predicted_result'].apply(lambda x: 'Home' if x == 1 else ('Away' if x == -1 else 'Draw'))
results = results.merge(teams[['team_id', 'team_name']], left_on='home_team_id', right_on='team_id', how='left')\
                 .rename(columns={'team_name': 'home_team_name'}).drop('team_id', axis=1)
results = results.merge(teams[['team_id', 'team_name']], left_on='away_team_id', right_on='team_id', how='left')\
                 .rename(columns={'team_name': 'away_team_name'}).drop('team_id', axis=1)
results = results[['fixture_date', 'time', 'match_id', 'home_team_name', 'away_team_name', 'winner', 
                  'home_win_prob', 'draw_prob', 'away_win_prob']]

# Round probabilities to 2 decimal places
results[['home_win_prob', 'draw_prob', 'away_win_prob']] = results[['home_win_prob', 'draw_prob', 'away_win_prob']].round(2)

# 10. Display Sample Predictions
print("\nSample Upcoming Match Predictions:")
print(results.head(5))

# 11. Save to PostgreSQL
create_table_query = """
CREATE TABLE IF NOT EXISTS match_predictions (
    fixture_date DATE,
    time TIME,
    match_id INTEGER,
    home_team_name VARCHAR(50),
    away_team_name VARCHAR(50),
    winner VARCHAR(10),
    home_win_prob FLOAT,
    draw_prob FLOAT,
    away_win_prob FLOAT
);
""" 