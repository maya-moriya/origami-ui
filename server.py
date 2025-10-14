from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from origami import Origami

app = Flask(__name__)
CORS(app)

# Global origami instance and history for undo functionality
origami = Origami(size=10.0)
history = []

def save_state():
    """Save current origami state to history for undo functionality"""
    state = {
        'vertices': {k: v.tolist() for k, v in origami.vertices.items()},
        'faces': origami.faces.copy(),
        'faces_orientations': origami.faces_orientations.copy(),
        'layers': origami.layers.copy(),
        'edges_splits': origami.edges_splits.copy(),
        'actions': origami.actions.copy()
    }
    history.append(json.loads(json.dumps(state)))  # Deep copy

def restore_state(state):
    """Restore origami state from history"""
    import numpy as np
    origami.vertices = {k: np.array(v) for k, v in state['vertices'].items()}
    origami.faces = state['faces']
    origami.faces_orientations = state['faces_orientations']
    origami.layers = state['layers']
    origami.edges_splits = state['edges_splits']
    origami.actions = state['actions']

def get_origami_data():
    """Convert origami state to JSON-serializable format"""
    return {
        'vertices': {k: v.tolist() for k, v in origami.vertices.items()},
        'faces': origami.faces,
        'faces_orientations': origami.faces_orientations,
        'layers': origami.layers,
        'edges_splits': origami.edges_splits,
        'actions': origami.actions,
        'all_edges': list(origami._get_all_edges())
    }

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/app.js')
def app_js():
    return send_from_directory('.', 'app.js')

@app.route('/api/state', methods=['GET'])
def get_state():
    """Get current origami state"""
    return jsonify(get_origami_data())

@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset origami to initial state"""
    global origami, history
    origami = Origami(size=10.0)
    history = []
    return jsonify({
        'success': True,
        **get_origami_data()
    })

@app.route('/api/split', methods=['POST'])
def split_edge():
    """Split an edge at the midpoint"""
    data = request.json
    edge = tuple(data['edge'])
    ratio = data.get('ratio', 0.5)
    
    save_state()
    try:
        new_vid = origami.split_face_by_edge_and_ratio(edge, ratio)
        return jsonify({
            'success': True,
            'new_vertex': new_vid,
            'state': get_origami_data()
        })
    except Exception as e:
        # Restore previous state on error
        if history:
            restore_state(history.pop())
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/fold', methods=['POST'])
def fold():
    """Fold along a crease line"""
    data = request.json
    edge = tuple(data['edge'])
    vertex_to_fold = int(data['vertex_to_fold'])  # Ensure it's an integer
    
    save_state()
    try:
        origami.fold_on_crease(edge, vertex_to_fold)
        return jsonify({
            'success': True,
            'state': get_origami_data()
        })
    except Exception as e:
        # Restore previous state on error
        if history:
            restore_state(history.pop())
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/flip', methods=['POST'])
def flip():
    """Flip the origami paper"""
    save_state()
    try:
        origami.flip()
        return jsonify({
            'success': True,
            'state': get_origami_data()
        })
    except Exception as e:
        # Restore previous state on error
        if history:
            restore_state(history.pop())
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/undo', methods=['POST'])
def undo():
    """Undo last operation"""
    if not history:
        return jsonify({
            'success': False,
            'error': 'Nothing to undo'
        })
    
    try:
        previous_state = history.pop()
        restore_state(previous_state)
        return jsonify({
            'success': True,
            'state': get_origami_data()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000)