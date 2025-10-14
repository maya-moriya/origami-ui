# Interactive Origami Simulator UI

A web-based user interface for the origami simulator that allows interactive folding and manipulation of origami paper.

## Features

- **Visual Interface**: Canvas-based display showing the origami paper with proper layering and colors
- **Interactive Vertices**: Indexed vertices (1, 2, 3, 4) that are clickable and draggable
- **Split Edge Mode**: Click on any edge to split it at the midpoint, creating new vertices
- **Fold Mode**: Drag between two vertices to define a crease line, then click a third vertex to fold
- **Flip Function**: Flip the entire paper over
- **Undo System**: Undo previous operations with full state restoration
- **Real-time Updates**: Canvas updates immediately after each operation

## Installation

1. Activate the conda environment:
```bash
conda activate origami-ui
```

2. Install Python dependencies (if not already installed):
```bash
pip install Flask Flask-CORS numpy matplotlib
```

## Running the Application

1. Make sure you're in the origami-ui conda environment:
```bash
conda activate origami-ui
```

2. Start the Flask server:
```bash
python server.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

### Split Edge Mode
1. Click the "Split Edge Mode" button
2. Click on any edge to split it at the midpoint
3. A new vertex will be created automatically

### Fold Mode
1. Click the "Fold Mode" button
2. Drag from one vertex to another to define the fold line (shown in red)
3. Click on a third vertex to choose which side of the paper to fold
4. The paper will fold along the selected line

### Other Controls
- **Flip Paper**: Flips the entire origami over
- **Undo**: Reverses the last operation

## File Structure

```
origami-ui/
├── src/
│   ├── origami.py       # Main origami simulator logic
│   ├── crease_map.py    # Crease pattern utilities
│   └── utils.py         # Mathematical utilities
├── index.html           # Main UI HTML
├── app.js              # JavaScript application logic
├── server.py           # Flask web server
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Technical Details

- **Backend**: Python Flask API serving the origami simulator
- **Frontend**: HTML5 Canvas with JavaScript for interactive UI
- **Communication**: RESTful API endpoints for operations
- **State Management**: Server-side state with history for undo functionality

## API Endpoints

- `GET /api/state` - Get current origami state
- `POST /api/split` - Split an edge
- `POST /api/fold` - Fold along a crease
- `POST /api/flip` - Flip the paper
- `POST /api/undo` - Undo last operation
- `POST /api/reset` - Reset to initial state