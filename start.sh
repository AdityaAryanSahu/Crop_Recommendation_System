#!/bin/bash
# start.sh

# 1. Add the my_app folder to the Python Path so Gunicorn can find it
export PYTHONPATH=$PYTHONPATH:./my_app

# 2. Run Gunicorn using the simplified import path
# The app is now fully importable via my_app.app.main:app
exec gunicorn my_app.app.main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$APP_PORT