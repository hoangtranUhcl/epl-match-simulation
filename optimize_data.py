import pandas as pd

# Optimize match results
match_results = pd.read_csv('match_results.csv')
match_results.to_csv('match_results_optimized.csv', index=False)

# Optimize player values
player_values = pd.read_csv('player_values.csv')
# Keep only necessary columns
essential_columns = ['player_id', 'market_value']  # adjust these columns based on what you actually use
player_values = player_values[essential_columns]
player_values.to_csv('player_values_optimized.csv', index=False)

# Create a smaller version of star players
star_players = pd.read_csv('star_players_by_team.csv')
star_players.to_csv('star_players_optimized.csv', index=False) 