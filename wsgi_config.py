import sys
import os

# Add your project directory to the sys.path
path = '/home/YOUR_USERNAME/epl-match-simulation'
if path not in sys.path:
    sys.path.append(path)

# Point to your virtual environment
os.environ['VIRTUAL_ENV'] = '/home/YOUR_USERNAME/.virtualenvs/epl-match-simulation'
os.environ['PATH'] = '/home/YOUR_USERNAME/.virtualenvs/epl-match-simulation/bin:' + os.environ['PATH']

# Import your Flask app
from app import app as application

# Optional: Set up logging
import logging
logging.basicConfig(stream=sys.stderr) 