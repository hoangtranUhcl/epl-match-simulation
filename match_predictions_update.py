import pandas as pd
from psycopg2 import Error
import psycopg2

# Database configuration
db_params = {
    "host": "68.74.165.203",
    "port": "5432",
    "database": "EPL",
    "user": "postgres",
    "password": "7410"
}

try:
    # Connect to PostgreSQL database
    connection = psycopg2.connect(**db_params)
    cursor = connection.cursor()

    # Read the match_predictions_fix file
    df = pd.read_csv('updated_match_predictions.csv')  # Added .csv extension

    print("DataFrame loaded:")
    print(df.head())

    # Create temporary table
    create_temp_table_query = """
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
    """
    cursor.execute(create_temp_table_query)

    # Insert data into temporary table
    for index, row in df.iterrows():
        insert_query = """
        INSERT INTO temp_match_prediction 
        (match_id, fixture_date, time, home_team_name, away_team_name, winner, home_win_prob, draw_prob, away_win_prob)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        cursor.execute(insert_query, (
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

    # First, delete any existing records that match the temp table
    delete_query = """
    DELETE FROM match_predictions 
    WHERE match_id IN (SELECT match_id FROM temp_match_prediction);
    """
    cursor.execute(delete_query)

    # Then insert all records from temp table
    insert_query = """
    INSERT INTO match_predictions 
    (match_id, fixture_date, time, home_team_name, away_team_name, winner, home_win_prob, draw_prob, away_win_prob)
    SELECT match_id, fixture_date, time, home_team_name, away_team_name, winner, home_win_prob, draw_prob, away_win_prob
    FROM temp_match_prediction;
    """
    cursor.execute(insert_query)

    # Drop temporary table
    drop_temp_table_query = "DROP TABLE temp_match_prediction;"
    cursor.execute(drop_temp_table_query)

    # Commit transaction
    connection.commit()
    print("Match predictions updated successfully!")

except FileNotFoundError:
    print("Error: match_predictions_fix.csv file not found!")
except (Exception, Error) as error:
    print("Error while updating match predictions:", error)
    if connection:
        connection.rollback()

finally:
    # Close database connection
    if connection:
        cursor.close()
        connection.close()
        print("PostgreSQL connection closed.")
