# GeoRoad Intelligence Interactive Web Studio

## 1. Overview
A full-stack web application designed for real-time interactive road network extraction from uploaded aerial/satellite imagery.

## 2. Architecture
- **Backend (`backend/main.py`)**: Asynchronous FastAPI service running on Uvicorn (Port 8000). Preloads the trained PyTorch model onto GPU VRAM, provides `/health` and `/extract-roads` endpoints, and streams Base64 rendered overlays with zero server disk I/O.
- **Frontend Standalone (`frontend_standalone/index.html`)**: Zero-install React 18 + Babel interface with dark glassmorphism design.
- **Production Frontend (`frontend/`)**: Vite + React template with modular component architecture.

## 3. Key Features
- **Canvas Modes**:
  - `512x512 Crop (Native)`: Native 1.0 m/px scale inspection.
  - `Full Canvas (Auto)`: Arbitrary dimension resizing with dynamic coordinate remapping ($g_x = x \cdot W/512$).
- **Interactive Threshold Sliders**:
  - `Road Threshold` (0.10 to 0.60): Controls centerline peak sensitivity.
  - `Intersection Threshold` (0.05 to 0.40): Controls junction sensitivity.
- **Multi-Tab Visualization**:
  - Vector Network Overlay (Emerald green road edges, crimson red intersection vertices).
  - Continuous Road Probability Heatmap (OpenCV JET colormap).
  - Topology JSON preview.
- **One-Click JSON Export**: Download `road_network_<timestamp>.json` with complete node coordinates and edge adjacency lists.

## 4. Quickstart
On Windows, simply double-click:
```cmd
start_app.bat
```
Or manually launch the backend:
```bash
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
Then navigate to: `http://127.0.0.1:8000/app`
