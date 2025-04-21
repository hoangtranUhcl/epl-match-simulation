import sys
path = '/home/timhoang/epl-match-simulation'
if path not in sys.path:
    sys.path.append(path)

from wsgi import application

if __name__ == '__main__':
    application.run() 