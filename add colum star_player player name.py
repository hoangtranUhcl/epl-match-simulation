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

    # Add player_name column if it doesn't exist
    add_column_query = """
    ALTER TABLE star_players_by_team
    ADD COLUMN IF NOT EXISTS player_name VARCHAR(100);
    """
    cursor.execute(add_column_query)

    # Update star_players_by_team with player names from player_data
    update_query = """
    UPDATE star_players_by_team
    SET player_name = player_data.player_name
    FROM player_data
    WHERE star_players_by_team.player_id = player_data.player_id;
    """
    cursor.execute(update_query)

    # Commit transaction
    connection.commit()
    print("Player names added successfully!")

except (Exception, Error) as error:
    print("Error while updating player names:", error)
    if connection:
        connection.rollback()

finally:
    # Close database connection
    if connection:
        cursor.close()
        connection.close()
        print("PostgreSQL connection closed.")
