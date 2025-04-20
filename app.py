from flask import Flask, render_template, request, jsonify
from match_prediction_manual import get_available_teams, get_star_players, predict_match
import logging
import os
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/')
def index():
    """Render the main page"""
    try:
        teams = get_available_teams()
        return render_template('index.html', teams=teams)
    except Exception as e:
        logger.error(f"Error loading teams: {str(e)}")
        logger.error(traceback.format_exc())
        return render_template('error.html', error="Failed to load teams. Please try again later.")

@app.route('/get_star_players/<team_name>')
def get_team_star_players(team_name):
    """Get star players for a team"""
    try:
        players = get_star_players(team_name)
        if not players:
            logger.warning(f"No star players found for {team_name}")
            return jsonify({'error': f'No star players found for {team_name}'}), 404
        return jsonify({
            'players': [{'id': str(p[0]), 'name': p[1], 'value': float(p[2])} for p in players]
        })
    except Exception as e:
        logger.error(f"Error getting star players for {team_name}: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Make a match prediction"""
    try:
        data = request.json
        logger.info(f"Received prediction request: {data}")

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Validate required fields
        required_fields = ['home_team', 'away_team']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400

        # Get excluded players - ensure they are integers
        excluded_players = []
        if 'excluded_players' in data and data['excluded_players']:
            try:
                excluded_players = [int(player_id) for player_id in data['excluded_players']]
                logger.info(f"Processing excluded players: {excluded_players}")
            except ValueError as ve:
                logger.error(f"Invalid player ID format: {data['excluded_players']}")
                return jsonify({'error': 'Invalid player ID format'}), 400

        # Make prediction
        logger.info(f"Making prediction for {data['home_team']} vs {data['away_team']} with excluded players: {excluded_players}")
        result = predict_match(
            data['home_team'],
            data['away_team'],
            excluded_players
        )
        logger.info(f"Prediction result: {result}")

        if not result:
            logger.error("Prediction returned None")
            return jsonify({'error': 'Failed to generate prediction'}), 500

        return jsonify(result)
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'details': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port) 