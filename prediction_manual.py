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

def inspect_table_structure():
    """Inspect the structure of required tables"""
    connection = None
    try:
        connection = psycopg2.connect(**db_params)
        cursor = connection.cursor()
        
        # Get column names for player_values table
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'player_values';
        """)
        columns = cursor.fetchall()
        print("Columns in player_values table:", [col[0] for col in columns])
        
    except (Exception, Error) as error:
        print("Error inspecting table structure:", error)
    finally:
        if connection:
            connection.close()

def get_db_connection():
    """Create and return a database connection"""
    try:
        return psycopg2.connect(**db_params)
    except Error as e:
        print(f"Error connecting to database: {e}")
        raise

def get_star_players(team_name):
    """Get star players for a team"""
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
        SELECT DISTINCT p.player_name, pv.market_value, pv.player_id
        FROM player_values pv
        JOIN player_data p ON pv.player_id = p.player_id
        WHERE pv.team_id = %s
        AND pv.season = (SELECT MAX(season) FROM player_values)
        ORDER BY pv.market_value DESC
        LIMIT 5
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
        SELECT team_id, player_id, market_value 
        FROM player_values
        WHERE team_id IN (%s, %s)
        AND season = (SELECT MAX(season) FROM player_values)
        """
        cursor.execute(player_query, (home_team_id, away_team_id))
        players = pd.DataFrame(cursor.fetchall(), 
                             columns=['team_id', 'player_id', 'market_value'])

        # Exclude unavailable players by player_id
        if excluded_players:
            players = players[~players['player_id'].isin(excluded_players)]

        # Calculate team values
        home_value = players[players['team_id'] == home_team_id]['market_value'].sum()
        away_value = players[players['team_id'] == away_team_id]['market_value'].sum()

        # Debugging output
        print("Home Value after excluding players:", home_value)
        print("Away Value after excluding players:", away_value)

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

        # Calculate performance metrics with default values if no data
        home_goals_scored = home_performance[home_performance['home_team_id'] == home_team_id]['home_goals'].mean() if not home_performance.empty else 1.5
        home_goals_conceded = home_performance[home_performance['away_team_id'] == home_team_id]['away_goals'].mean() if not home_performance.empty else 1.0
        away_goals_scored = away_performance[away_performance['away_team_id'] == away_team_id]['away_goals'].mean() if not away_performance.empty else 1.5
        away_goals_conceded = away_performance[away_performance['home_team_id'] == away_team_id]['home_goals'].mean() if not away_performance.empty else 1.0

        return {
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'home_value': float(home_value),
            'away_value': float(away_value),
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
        
        # Create feature vector
        features = pd.DataFrame([[
            team_data['home_value'],
            team_data['away_value'],
            team_data['home_goals_scored'],
            team_data['home_goals_conceded'],
            team_data['away_goals_scored'],
            team_data['away_goals_conceded']
        ]], columns=['home_value', 'away_value', 'home_goals_scored', 'home_goals_conceded', 
                     'away_goals_scored', 'away_goals_conceded'])

        print("Feature vector for prediction:", features)

        # Train model on historical data
        connection = get_db_connection()
        cursor = connection.cursor()

        # Get historical matches with team values
        hist_query = """
        WITH team_values AS (
            SELECT 
                team_id,
                season,
                SUM(market_value) as team_value
            FROM player_values
            GROUP BY team_id, season
        )
        SELECT 
            hv.team_value as home_team_value,
            av.team_value as away_team_value,
            m.home_goals,
            m.away_goals,
            m.result
        FROM match_results m
        JOIN team_values hv ON m.home_team_id = hv.team_id
        JOIN team_values av ON m.away_team_id = av.team_id
        WHERE m.result IS NOT NULL
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

        # Train model
        le = LabelEncoder()
        y = le.fit_transform(y)  # Convert to 0,1,2
        
        # Calculate class weights
        class_counts = np.bincount(y)
        total_samples = len(y)
        class_weights = {i: total_samples / (len(class_counts) * count) if count > 0 else 1.0 for i, count in enumerate(class_counts)}
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,  # Increased for complexity
            min_samples_split=5,  # Adjusted for flexibility
            min_samples_leaf=2,  # Adjusted for responsiveness
            class_weight=class_weights,
            random_state=42
        )
        model.fit(X_scaled, y)

        # Make prediction
        raw_probabilities = model.predict_proba(features_scaled)[0]
        print("Raw probabilities before scaling:", raw_probabilities)

        temperature = 1.2  # Adjusted for sharper probabilities
        scaled_probabilities = np.exp(np.log(raw_probabilities) / temperature)
        scaled_probabilities = scaled_probabilities / np.sum(scaled_probabilities)
        probabilities = scaled_probabilities * 100

        # Get predicted class
        prediction = np.argmax(probabilities)
        
        # Map prediction index to result label
        result_mapping = {i: result for i, result in enumerate(le.classes_)}
        predicted_result = result_mapping.get(prediction, "Unknown")

        result = {
            'predicted_result': predicted_result,
            'home_win_prob': round(probabilities[le.transform(['Home Win'])[0] if 'Home Win' in le.classes_ else 0], 2),
            'draw_prob': round(probabilities[le.transform(['Draw'])[0] if 'Draw' in le.classes_ else 1], 2),
            'away_win_prob': round(probabilities[le.transform(['Away Win'])[0] if 'Away Win' in le.classes_ else 2], 2),
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

# Example usage:
if __name__ == "__main__":
    try:
        print("Inspecting table structure...")
        inspect_table_structure()
        
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
            for i, (name, value, player_id) in enumerate(home_stars, 1):
                print(f"{i}. {name} (Market Value: £{value:,.0f})")
        else:
            print("No star players found in database.")
        
        print(f"\nStar players for {away_team}:")
        away_stars = get_star_players(away_team)
        if away_stars:
            for i, (name, value, player_id) in enumerate(away_stars, 1):
                print(f"{i}. {name} (Market Value: £{value:,.0f})")
        else:
            print("No star players found in database.")
        
        # Get number of unavailable players for home team
        while True:
            try:
                num_home_unavailable = int(input(f"\nEnter number of unavailable players for {home_team} (0-5): "))
                if 0 <= num_home_unavailable <= 5:
                    break
                print("Please enter a number between 0 and 5.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        # Get unavailable players for home team - NOW STORES PLAYER IDs NOT NAMES
        home_excluded_players = []
        if num_home_unavailable > 0 and home_stars:
            print(f"\nSelect unavailable players for {home_team} (enter numbers separated by spaces):")
            for i, (name, _, _) in enumerate(home_stars, 1):
                print(f"{i}. {name}")
            
            while len(home_excluded_players) < num_home_unavailable:
                try:
                    selection = input(f"\nEnter player number ({len(home_excluded_players) + 1}/{num_home_unavailable}): ")
                    player_idx = int(selection) - 1
                    if 0 <= player_idx < len(home_stars):
                        # Store player ID, not name
                        home_excluded_players.append(home_stars[player_idx][2])  # player_id is the 3rd element in tuple
                    else:
                        print("Invalid player number. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
        elif num_home_unavailable > 0:
            print(f"No star players available to exclude for {home_team}")
        
        # Get number of unavailable players for away team
        while True:
            try:
                num_away_unavailable = int(input(f"\nEnter number of unavailable players for {away_team} (0-5): "))
                if 0 <= num_away_unavailable <= 5:
                    break
                print("Please enter a number between 0 and 5.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        # Get unavailable players for away team - NOW STORES PLAYER IDs NOT NAMES
        away_excluded_players = []
        if num_away_unavailable > 0 and away_stars:
            print(f"\nSelect unavailable players for {away_team} (enter numbers separated by spaces):")
            for i, (name, _, _) in enumerate(away_stars, 1):
                print(f"{i}. {name}")
            
            while len(away_excluded_players) < num_away_unavailable:
                try:
                    selection = input(f"\nEnter player number ({len(away_excluded_players) + 1}/{num_away_unavailable}): ")
                    player_idx = int(selection) - 1
                    if 0 <= player_idx < len(away_stars):
                        # Store player ID, not name
                        away_excluded_players.append(away_stars[player_idx][2])  # player_id is the 3rd element in tuple
                    else:
                        print("Invalid player number. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
        elif num_away_unavailable > 0:
            print(f"No star players available to exclude for {away_team}")
        
        # Combine excluded players
        excluded_players = home_excluded_players + away_excluded_players
        
        # For display purposes, we'll keep track of player names separately
        home_excluded_player_names = [home_stars[home_stars.index(next(p for p in home_stars if p[2] == player_id))][0] 
                                     for player_id in home_excluded_players] if home_excluded_players else []
        away_excluded_player_names = [away_stars[away_stars.index(next(p for p in away_stars if p[2] == player_id))][0] 
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
        print("3. The required tables exist (teams, player_values, match_results)")