@echo off
title GeoRoad Intelligence - FastAPI & React Web App
echo ========================================================
echo   Launching GeoRoad Intelligence
echo   FastAPI Backend + React Frontend
echo   Model Checkpoint: FinalModel\checkpoints\best_road_model.pth
echo ========================================================
echo.

cd /d "%~dp0backend"
echo Starting FastAPI Web Server at http://127.0.0.1:8000/app ...
echo Open your browser at: http://127.0.0.1:8000/app
echo API Documentation at: http://127.0.0.1:8000/docs
echo.

python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
pause

