# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Activate the conda environment
conda activate origami-ui

# Install Python dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Start the Flask development server
python server.py

# The application will be available at http://localhost:5000
```

### Python Development
```bash
# Run Python code with proper path setup
python -c "import sys; sys.path.append('src'); from origami import Origami; print('Success')"
```

## Architecture Overview

### Project Structure
- **Backend**: Python Flask API with origami simulation engine
- **Frontend**: Vanilla JavaScript with HTML5 Canvas for interactive UI
- **Communication**: RESTful API endpoints for all operations

### Core Components

#### Python Backend (`src/`)
- **`origami.py`**: Main origami simulation engine
  - `Origami` class handles paper state, folding operations, and geometry
  - Manages vertices, faces, layers, and orientations
  - Core methods: `split_face_by_edge_and_ratio()`, `fold_on_crease()`, `flip()`

- **`utils.py`**: Mathematical utilities
  - Line equations and geometric calculations
  - Point reflection and intersection functions
  - Face overlap detection using Separating Axis Theorem (SAT)

- **`crease_map.py`**: Crease pattern analysis
  - Valley/mountain fold classification
  - Edge-face relationship mapping
  - Vertex position recovery for visualization

#### Frontend
- **`app.js`**: JavaScript UI controller
  - `OrigamiUI` class manages canvas rendering and user interactions
  - Coordinate transformation between world and canvas space
  - Real-time visual feedback for edge splitting and folding
  - State management for fold operations

- **`index.html`**: Single-page application with canvas and controls
- **`server.py`**: Flask web server with CORS support

### Key Architecture Patterns

#### State Management
- Server maintains authoritative origami state
- History system for undo functionality using deep state copies
- Client synchronizes with server after each operation

#### Coordinate Systems
- Backend uses mathematical coordinate system (Y-up)
- Frontend canvas uses screen coordinates (Y-down) 
- Transformation functions handle conversion between systems

#### Layer System
- Faces are organized into layers for proper rendering order
- Layer assignment prevents overlapping faces
- Orientation tracking (0=white, 1=lightblue) for visual distinction

#### Interactive Operations
1. **Edge Splitting**: Click on edges to add vertices at specific ratios
2. **Folding**: Drag between vertices to define fold line, click third vertex to fold
3. **Special Ratios**: UI snaps to common ratios (1/3, 1/2, 2/3, etc.)

### API Endpoints
- `GET /api/state` - Get current origami state
- `POST /api/split` - Split edge at specified ratio
- `POST /api/fold` - Fold along crease line
- `POST /api/flip` - Flip entire paper
- `POST /api/undo` - Undo last operation
- `POST /api/reset` - Reset to initial square

### Development Notes
- Debug mode enabled in origami.py with detailed logging
- Canvas uses static scaling (50% of screen size) for consistent coordinate space
- Vertex grouping handles overlapping vertices at same position
- Error handling restores previous state on operation failure