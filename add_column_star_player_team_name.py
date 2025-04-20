import psycopg2

# Database configuration
db_params = {
    "host": "68.74.165.203",
    "port": "5432",
    "database": "EPL",
    "user": "postgres",
    "password": "7410"
}

# Connect to the PostgreSQL database
connection = psycopg2.connect(**db_params)
cursor = connection.cursor()

try:
    # Step 1: Check if the team_name column exists
    cursor.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name='star_players_by_team' AND column_name='team_name';
    """)
    
    # If the column does not exist, add it
    if not cursor.fetchone():
        cursor.execute("""
            ALTER TABLE star_players_by_team
            ADD COLUMN team_name VARCHAR(50);
        """)
        print("team_name column added.")
    else:
        print("team_name column already exists.")

    # Step 2: Update the team_name column based on team_id
    cursor.execute("""
        UPDATE star_players_by_team AS sp
        SET team_name = t.team_name
        FROM teams AS t
        WHERE sp.team_id = t.team_id;  -- Specify the table for team_id
    """)

    # Step 3: Create a new table with the desired column order
    cursor.execute("""
        CREATE TABLE new_star_players_by_team (
            player_value_id SERIAL PRIMARY KEY,
            player_id INTEGER,
            team_id INTEGER,
            team_name VARCHAR(50),
            market_value NUMERIC,
            season VARCHAR(10),
            rank_in_team INTEGER,
            player_name VARCHAR(100)
        );
    """)

    # Step 4: Copy data from the old table to the new table
    cursor.execute("""
        INSERT INTO new_star_players_by_team (player_value_id, player_id, team_id, market_value, season, rank_in_team, player_name, team_name)
        SELECT sp.player_value_id, sp.player_id, sp.team_id, sp.market_value, sp.season, sp.rank_in_team, sp.player_name, t.team_name
        FROM star_players_by_team AS sp
        LEFT JOIN teams AS t ON sp.team_id = t.team_id;
    """)

    # Step 5: Drop the old table
    cursor.execute("DROP TABLE star_players_by_team;")

    # Step 6: Rename the new table to the original name
    cursor.execute("ALTER TABLE new_star_players_by_team RENAME TO star_players_by_team;")

    # Commit the changes
    connection.commit()
    print("team_name column updated successfully.")

except Exception as e:
    print(f"An error occurred: {e}")
    connection.rollback()

finally:
    # Close the database connection
    cursor.close()
    connection.close()