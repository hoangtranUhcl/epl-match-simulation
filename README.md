# Match Prediction System

A Flask-based web application for predicting football match outcomes based on team values and player availability.

## Features

- Select home and away teams
- View star players for each team
- Mark players as unavailable
- Get match predictions with probabilities
- Visual representation of prediction results

## Prerequisites

- Python 3.7 or higher
- PostgreSQL database
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd match-prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure the database:
- Update the database configuration in `match_prediction_manual.py` with your PostgreSQL credentials:
```python
db_params = {
    "host": "your_host",
    "port": "your_port",
    "database": "your_database",
    "user": "your_username",
    "password": "your_password"
}
```

## Running the Application

1. Start the Flask development server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

## Usage

1. Select a home team from the dropdown menu
2. Select an away team from the dropdown menu
3. View available star players for each team
4. Mark any unavailable players by checking the boxes
5. Click "Predict Match" to get the prediction
6. View the prediction results with probabilities

## Project Structure

- `app.py` - Flask application and routes
- `match_prediction_manual.py` - Core prediction logic
- `templates/` - HTML templates
  - `index.html` - Main application interface
  - `error.html` - Error page template
- `requirements.txt` - Python dependencies

## Error Handling

The application includes error handling for:
- Database connection issues
- Invalid team selections
- Missing required fields
- Server errors

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 