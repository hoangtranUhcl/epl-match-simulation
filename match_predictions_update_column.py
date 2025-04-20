import pandas as pd

# Load data from CSV
fixtures = pd.read_csv('match_predictions.csv')

# Combine fixture_date and time into a new column fixture_datetime
fixtures['fixture_datetime'] = pd.to_datetime(fixtures['fixture_date'] + ' ' + fixtures['time'])

# Display the updated DataFrame to verify the new column
print(fixtures[['fixture_date', 'time', 'fixture_datetime']].head())

# Save the updated DataFrame to a new CSV file
fixtures.to_csv('updated_match_predictions.csv', index=False)