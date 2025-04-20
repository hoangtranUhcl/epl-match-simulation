import pandas as pd
import psycopg2
from psycopg2 import Error

# Database connection parameters
db_params = {
    "host": "68.74.165.203",
    "port": "5432",
    "database": "EPL",
    "user": "postgres",
    "password": "7410"
}

# Path to your CSV file
csv_file_path = "match_results.csv"

# Step 1: Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file_path)

# Step 2: Display the first few rows of the DataFrame
print("First few rows of the DataFrame:")
print(df.head())

# Step 3: Check the data types of the columns
print("\nData types of the columns:")
print(df.dtypes)

# Step 4: Check for maximum values in integer columns
integer_columns = ['match_id', 'home_goals', 'away_goals', 'result']  # Adjust based on your actual column names

for column in integer_columns:
    if column in df.columns:
        max_value = df[column].max()
        print(f"Maximum value in '{column}': {max_value}")
        if max_value > 2147483647:  # Check against the maximum limit for INTEGER
            print(f"Warning: '{column}' exceeds the maximum integer limit.")

# Step 5: Check for any non-integer values in integer columns
for column in integer_columns:
    if column in df.columns:
        non_integer_values = df[~df[column].apply(lambda x: isinstance(x, (int, float)))]
        if not non_integer_values.empty:
            print(f"Non-integer values found in '{column}':")
            print(non_integer_values)

# Step 6: Check for any NaN values in integer columns
for column in integer_columns:
    if column in df.columns:
        if df[column].isnull().any():
            print(f"Warning: '{column}' contains NaN values.")

# Convert date column to a proper date format (assuming MM/DD/YY format in CSV)
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%y').dt.strftime('%Y-%m-%d')

# Step 2: Connect to PostgreSQL and update the table
try:
    # Establish connection
    connection = psycopg2.connect(**db_params)
    cursor = connection.cursor()

    # Step 3: Create a temporary table
    create_temp_table_query = """
    CREATE TABLE temp_match_result (
        match_id INTEGER,
        date DATE,
        home_team_id INTEGER,
        away_team_id INTEGER,
        home_goals INTEGER,
        away_goals INTEGER,
        result INTEGER
    );
    """
    cursor.execute(create_temp_table_query)

    # Step 4: Insert data into the temporary table
    for index, row in df.iterrows():
        insert_query = """
        INSERT INTO temp_match_result (match_id, date, home_team_id, away_team_id, home_goals, away_goals, result)
        VALUES (%s, %s, %s, %s, %s, %s, %s);
        """
        cursor.execute(insert_query, (
            row['match_id'], 
            row['date'], 
            row['home_team_id'], 
            row['away_team_id'], 
            row['home_goals'], 
            row['away_goals'], 
            row['result']
        ))

    # Step 5: Update the main table from the temporary table
    update_query = """
    UPDATE match_results
    SET 
        date = temp_match_result.date,
        home_team_id = temp_match_result.home_team_id,
        away_team_id = temp_match_result.away_team_id,
        home_goals = temp_match_result.home_goals,
        away_goals = temp_match_result.away_goals,
        result = temp_match_result.result
    FROM temp_match_result
    WHERE match_results.match_id = temp_match_result.match_id;
    """
    cursor.execute(update_query)

    # Step 6: Insert new records that don't exist in the main table
    insert_new_query = """
    INSERT INTO match_results (match_id, date, home_team_id, away_team_id, home_goals, away_goals, result)
    SELECT match_id, date, home_team_id, away_team_id, home_goals, away_goals, result
    FROM temp_match_result
    WHERE NOT EXISTS (
        SELECT 1 FROM match_results WHERE match_results.match_id = temp_match_result.match_id
    );
    """
    cursor.execute(insert_new_query)

    # Step 7: Drop the temporary table
    drop_temp_table_query = "DROP TABLE temp_match_result;"
    cursor.execute(drop_temp_table_query)

    # Commit the transaction
    connection.commit()
    print("Database updated successfully!")

except (Exception, Error) as error:
    print("Error while connecting to PostgreSQL or updating data:", error)
    if connection:
        connection.rollback()

finally:
    # Close the database connection
    if connection:
        cursor.close()
        connection.close()
        print("PostgreSQL connection closed.")