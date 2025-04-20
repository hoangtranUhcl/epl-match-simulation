import pandas as pd
import psycopg2
from psycopg2 import Error
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# Database configuration
db_params = {
    "host": "68.74.165.203", 
    "port": "5432",
    "database": "EPL",
    "user": "postgres",
    "password": "7410"
}

def get_db_connection():
    """Create and return a database connection"""
    try:
        return psycopg2.connect(**db_params)
    except Error as e:
        print(f"Error connecting to database: {e}")
        raise

def get_available_teams():
    """Get list of available teams"""
    connection = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        cursor.execute("SELECT team_name FROM teams ORDER BY team_name")
        teams = [row[0] for row in cursor.fetchall()]
        return teams
    finally:
        if connection:
            connection.close()

def get_star_players(team_name):
    """Get star players for a team from star_players_by_team table"""
    connection = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        # Get team ID
        team_query = """
        SELECT team_id FROM teams WHERE team_name = %s
        """
        cursor.execute(team_query, (team_name,))
        team_id = cursor.fetchone()
        
        if not team_id:
            raise ValueError(f"Team not found: {team_name}")
        
        team_id = team_id[0]

        # Get star players with their market values
        player_query = """
        SELECT sp.player_id, sp.player_name, sp.market_value
        FROM star_players_by_team sp
        WHERE sp.team_id = %s
        ORDER BY sp.market_value DESC
        """
        cursor.execute(player_query, (team_id,))
        players = cursor.fetchall()
        
        return players

    except (Exception, Error) as error:
        print("Error while getting star players:", error)
        raise
    finally:
        if connection:
            connection.close()

def get_team_data(home_team, away_team, excluded_players=None):
    """Get team data including player values and recent performance"""
    connection = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        # Get team IDs
        team_query = """
        SELECT team_id, team_name FROM teams 
        WHERE team_name IN (%s, %s)
        """
        cursor.execute(team_query, (home_team, away_team))
        team_data = dict(cursor.fetchall())
        
        if not team_data:
            raise ValueError(f"Teams not found: {home_team} or {away_team}")
        
        home_team_id = [k for k,v in team_data.items() if v == home_team][0]
        away_team_id = [k for k,v in team_data.items() if v == away_team][0]

        # Get player values for both teams
        player_query = """
        SELECT sp.team_id, sp.player_id, sp.player_name, sp.market_value 
        FROM star_players_by_team sp
        WHERE sp.team_id IN (%s, %s)
        """
        cursor.execute(player_query, (home_team_id, away_team_id))
        players = pd.DataFrame(cursor.fetchall(), 
                             columns=['team_id', 'player_id', 'player_name', 'market_value'])

        # Convert market values to float
        players['market_value'] = players['market_value'].astype(float)

        # Calculate total team values before exclusions
        total_home_value = float(players[players['team_id'] == home_team_id]['market_value'].sum())
        total_away_value = float(players[players['team_id'] == away_team_id]['market_value'].sum())

        # Exclude unavailable players and calculate remaining values
        if excluded_players:
            excluded_players_df = players[players['player_id'].isin(excluded_players)]
            excluded_home_value = float(excluded_players_df[excluded_players_df['team_id'] == home_team_id]['market_value'].sum())
            excluded_away_value = float(excluded_players_df[excluded_players_df['team_id'] == away_team_id]['market_value'].sum())
            
            home_value = total_home_value - excluded_home_value
            away_value = total_away_value - excluded_away_value
            
            # Calculate impact of unavailable players
            home_impact = float(excluded_home_value / total_home_value if total_home_value > 0 else 0)
            away_impact = float(excluded_away_value / total_away_value if total_away_value > 0 else 0)

            # Print detailed information about excluded players
            print("\nExcluded Players Details:")
            for _, row in excluded_players_df.iterrows():
                team = home_team if row['team_id'] == home_team_id else away_team
                print(f"{row['player_name']} ({team}): £{float(row['market_value']):,.0f}")
        else:
            home_value = total_home_value
            away_value = total_away_value
            home_impact = 0.0
            away_impact = 0.0

        # Debugging output
        print(f"\nTeam Values:")
        print(f"Home Team ({home_team}):")
        print(f"  Total Value: £{total_home_value:,.0f}")
        print(f"  Available Value: £{home_value:,.0f}")
        print(f"  Impact of Unavailable Players: {home_impact:.2%}")
        print(f"Away Team ({away_team}):")
        print(f"  Total Value: £{total_away_value:,.0f}")
        print(f"  Available Value: £{away_value:,.0f}")
        print(f"  Impact of Unavailable Players: {away_impact:.2%}")

        # Get recent performance (last 5 matches)
        performance_query = """
        SELECT 
            home_team_id,
            away_team_id,
            home_goals,
            away_goals,
            result
        FROM match_results
        WHERE (home_team_id = %s OR away_team_id = %s)
        ORDER BY date DESC
        LIMIT 5
        """
        cursor.execute(performance_query, (home_team_id, home_team_id))
        home_performance = pd.DataFrame(cursor.fetchall(), 
                                     columns=['home_team_id', 'away_team_id', 'home_goals', 'away_goals', 'result'])
        
        cursor.execute(performance_query, (away_team_id, away_team_id))
        away_performance = pd.DataFrame(cursor.fetchall(), 
                                     columns=['home_team_id', 'away_team_id', 'home_goals', 'away_goals', 'result'])

        # Convert goals to float and calculate performance metrics with default values if no data
        home_goals_scored = float(home_performance[home_performance['home_team_id'] == home_team_id]['home_goals'].mean() if not home_performance.empty else 1.5)
        home_goals_conceded = float(home_performance[home_performance['away_team_id'] == home_team_id]['away_goals'].mean() if not home_performance.empty else 1.0)
        away_goals_scored = float(away_performance[away_performance['away_team_id'] == away_team_id]['away_goals'].mean() if not away_performance.empty else 1.5)
        away_goals_conceded = float(away_performance[away_performance['home_team_id'] == away_team_id]['home_goals'].mean() if not away_performance.empty else 1.0)

        # Adjust performance metrics based on player impact
        home_goals_scored *= (1.0 - home_impact)
        home_goals_conceded *= (1.0 + home_impact)
        away_goals_scored *= (1.0 - away_impact)
        away_goals_conceded *= (1.0 + away_impact)

        return {
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'home_value': float(home_value),
            'away_value': float(away_value),
            'home_impact': float(home_impact),
            'away_impact': float(away_impact),
            'home_goals_scored': float(home_goals_scored),
            'home_goals_conceded': float(home_goals_conceded),
            'away_goals_scored': float(away_goals_scored),
            'away_goals_conceded': float(away_goals_conceded)
        }

    except (Exception, Error) as error:
        print("Error while getting team data:", error)
        raise
    finally:
        if connection:
            connection.close()

def predict_match(home_team, away_team, excluded_players=None):
    """Predict match outcome with probabilities"""
    connection = None
    try:
        # Get team data
        team_data = get_team_data(home_team, away_team, excluded_players)
        
        # Create feature vector matching CurrentModel.py
        features = pd.DataFrame([[
            team_data['home_value'],
            team_data['away_value'],
            team_data['home_goals_scored'],
            team_data['home_goals_conceded'],
            team_data['away_goals_scored'],
            team_data['away_goals_conceded']
        ]], columns=['home_value', 'away_value', 'home_goals_scored', 'home_goals_conceded', 
                     'away_goals_scored', 'away_goals_conceded'])

        print("\nFeature vector for prediction:")
        print(features)

        # Train model on historical data
        connection = get_db_connection()
        cursor = connection.cursor()

        # Get historical matches with team values
        hist_query = """
        WITH team_values AS (
            SELECT 
                team_id,
                SUM(market_value) as team_value
            FROM star_players_by_team
            GROUP BY team_id
        )
        SELECT 
            hv.team_value as home_team_value,
            av.team_value as away_team_value,
            m.home_goals,
            m.away_goals,
            CASE 
                WHEN m.home_goals > m.away_goals THEN 2  -- Home Win
                WHEN m.home_goals < m.away_goals THEN 0  -- Away Win
                ELSE 1  -- Draw
            END as result
        FROM match_results m
        JOIN team_values hv ON m.home_team_id = hv.team_id
        JOIN team_values av ON m.away_team_id = av.team_id
        WHERE m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL
        """
        cursor.execute(hist_query)
        hist_data = pd.DataFrame(cursor.fetchall(), 
                               columns=['home_value', 'away_value', 'home_goals', 'away_goals', 'result'])

        if hist_data.empty:
            raise ValueError("No historical data available for training")

        # Prepare training data
        X = pd.DataFrame({
            'home_value': hist_data['home_value'].astype(float),
            'away_value': hist_data['away_value'].astype(float),
            'home_goals_scored': hist_data['home_goals'].astype(float),
            'home_goals_conceded': hist_data['away_goals'].astype(float),
            'away_goals_scored': hist_data['away_goals'].astype(float),
            'away_goals_conceded': hist_data['home_goals'].astype(float)
        })
        y = hist_data['result']

        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        features_scaled = scaler.transform(features)

        # Train model with XGBoost parameters from CurrentModel.py
        from xgboost import XGBClassifier
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42,
            n_estimators=150,
            learning_rate=0.05,
            max_depth=7,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1,
            gamma=0.5
        )
        model.fit(X_scaled, y)

        # Make prediction
        raw_probabilities = model.predict_proba(features_scaled)[0]
        print("\nRaw probabilities before scaling:", raw_probabilities)

        # Apply temperature scaling
        temperature = 1.2
        scaled_probabilities = np.exp(np.log(raw_probabilities) / temperature)
        scaled_probabilities = scaled_probabilities / np.sum(scaled_probabilities)
        probabilities = scaled_probabilities * 100

        # Get predicted class
        prediction = np.argmax(probabilities)
        
        # Map prediction index to result label
        result_mapping = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
        predicted_result = result_mapping.get(prediction, "Unknown")

        result = {
            'predicted_result': predicted_result,
            'home_win_prob': round(probabilities[2], 2),
            'draw_prob': round(probabilities[1], 2),
            'away_win_prob': round(probabilities[0], 2),
            'home_team': home_team,
            'away_team': away_team,
            'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        return result

    except (Exception, Error) as error:
        print("Error during prediction:", error)
        raise
    finally:
        if connection:
            connection.close()

def main():
    """Main execution function"""
    try:
        print("\nStarting prediction...")
        # Get list of available teams
        available_teams = get_available_teams()
        print("\nAvailable teams:")
        for i, team in enumerate(available_teams, 1):
            print(f"{i}. {team}")
        
        # Get user input for teams
        while True:
            try:
                home_team = input("\nEnter home team name: ")
                if home_team in available_teams:
                    break
                print("Invalid team name. Please try again.")
            except ValueError:
                print("Invalid input. Please try again.")
        
        while True:
            try:
                away_team = input("Enter away team name: ")
                if away_team in available_teams and away_team != home_team:
                    break
                print("Invalid team name or same as home team. Please try again.")
            except ValueError:
                print("Invalid input. Please try again.")
        
        # Get star players for both teams
        print(f"\nStar players for {home_team}:")
        home_stars = get_star_players(home_team)
        if home_stars:
            for i, (player_id, name, value) in enumerate(home_stars, 1):
                print(f"{i}. {name} (Market Value: £{value:,.0f})")
        else:
            print("No star players found in database.")
        
        print(f"\nStar players for {away_team}:")
        away_stars = get_star_players(away_team)
        if away_stars:
            for i, (player_id, name, value) in enumerate(away_stars, 1):
                print(f"{i}. {name} (Market Value: £{value:,.0f})")
        else:
            print("No star players found in database.")
        
        # Get number of unavailable players for home team
        while True:
            try:
                num_home_unavailable = int(input(f"\nEnter number of unavailable players for {home_team} (0-{len(home_stars)}): "))
                if 0 <= num_home_unavailable <= len(home_stars):
                    break
                print(f"Please enter a number between 0 and {len(home_stars)}.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        # Get unavailable players for home team
        home_excluded_players = []
        if num_home_unavailable > 0 and home_stars:
            print(f"\nSelect unavailable players for {home_team} (enter numbers separated by spaces):")
            for i, (_, name, _) in enumerate(home_stars, 1):
                print(f"{i}. {name}")
            
            while len(home_excluded_players) < num_home_unavailable:
                try:
                    selection = input(f"\nEnter player number ({len(home_excluded_players) + 1}/{num_home_unavailable}): ")
                    player_idx = int(selection) - 1
                    if 0 <= player_idx < len(home_stars):
                        home_excluded_players.append(home_stars[player_idx][0])  # player_id is the first element
                    else:
                        print("Invalid player number. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
        elif num_home_unavailable > 0:
            print(f"No star players available to exclude for {home_team}")
        
        # Get number of unavailable players for away team
        while True:
            try:
                num_away_unavailable = int(input(f"\nEnter number of unavailable players for {away_team} (0-{len(away_stars)}): "))
                if 0 <= num_away_unavailable <= len(away_stars):
                    break
                print(f"Please enter a number between 0 and {len(away_stars)}.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        # Get unavailable players for away team
        away_excluded_players = []
        if num_away_unavailable > 0 and away_stars:
            print(f"\nSelect unavailable players for {away_team} (enter numbers separated by spaces):")
            for i, (_, name, _) in enumerate(away_stars, 1):
                print(f"{i}. {name}")
            
            while len(away_excluded_players) < num_away_unavailable:
                try:
                    selection = input(f"\nEnter player number ({len(away_excluded_players) + 1}/{num_away_unavailable}): ")
                    player_idx = int(selection) - 1
                    if 0 <= player_idx < len(away_stars):
                        away_excluded_players.append(away_stars[player_idx][0])  # player_id is the first element
                    else:
                        print("Invalid player number. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
        elif num_away_unavailable > 0:
            print(f"No star players available to exclude for {away_team}")
        
        # Combine excluded players
        excluded_players = home_excluded_players + away_excluded_players
        
        # For display purposes, we'll keep track of player names separately
        home_excluded_player_names = [home_stars[home_stars.index(next(p for p in home_stars if p[0] == player_id))][1] 
                                     for player_id in home_excluded_players] if home_excluded_players else []
        away_excluded_player_names = [away_stars[away_stars.index(next(p for p in away_stars if p[0] == player_id))][1] 
                                     for player_id in away_excluded_players] if away_excluded_players else []
        
        # Make prediction
        print("\nMaking prediction...")
        result = predict_match(home_team, away_team, excluded_players)
        
        # Display results
        print(f"\nPrediction for {home_team} vs {away_team}")
        print(f"Prediction time: {result['prediction_time']}")
        if excluded_players:
            if home_excluded_player_names:
                print(f"Unavailable players for {home_team}: {', '.join(home_excluded_player_names)}")
            if away_excluded_player_names:
                print(f"Unavailable players for {away_team}: {', '.join(away_excluded_player_names)}")
        print(f"Predicted result: {result['predicted_result']}")
        print(f"Probabilities:")
        print(f"Home win: {result['home_win_prob']}%")
        print(f"Draw: {result['draw_prob']}%")
        print(f"Away win: {result['away_win_prob']}%")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        print("Please check if:")
        print("1. The database is accessible")
        print("2. The team names are correct")
        print("3. The required tables exist (teams, star_players_by_team, match_results)")

if __name__ == "__main__":
    main() 