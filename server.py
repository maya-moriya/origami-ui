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

@app.route('/icons/<filename>')
def serve_icon(filename):
    return send_from_directory('icons', filename)

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
    
    print(f"\n=== SPLIT REQUEST ===")
    print(f"Edge to split: {edge}")
    print(f"Ratio: {ratio}")
    print(f"Current vertices: {dict(origami.vertices)}")
    print(f"Current faces: {origami.faces}")
    print(f"All edges: {list(origami._get_all_edges())}")
    
    # Check if edge exists
    all_edges = list(origami._get_all_edges())
    if edge not in all_edges and tuple(reversed(edge)) not in all_edges:
        print(f"ERROR: Edge {edge} not found in all_edges!")
        return jsonify({
            'success': False,
            'error': f'Edge {edge} does not exist'
        })
    
    save_state()
    try:
        print(f"Attempting to split edge {edge} at ratio {ratio}")
        new_vid = origami.split_face_by_edge_and_ratio(edge, ratio)
        print(f"Split successful, new vertex ID: {new_vid}")
        print(f"New vertices: {dict(origami.vertices)}")
        return jsonify({
            'success': True,
            'new_vertex': new_vid,
            'state': get_origami_data()
        })
    except Exception as e:
        print(f"Split error: {type(e).__name__}: {str(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        
        # Restore previous state on error
        if history:
            restore_state(history.pop())
        return jsonify({
            'success': False,
            'error': f'{type(e).__name__}: {str(e)}'
        })

@app.route('/api/fold_options', methods=['POST'])
def fold_options():
    """Get fold options for a crease"""
    data = request.json
    edge = tuple(data['edge'])
    
    try:
        line, bunch_positive, bunch_negative, sorted_faces, face_layer_map, vertices_sides, vertices_faces = origami.sss_ui_pre_fold(edge)
        return jsonify({
            'success': True,
            'line': line,
            'faces_positive': list(bunch_positive),
            'faces_negative': list(bunch_negative)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/fold', methods=['POST'])
def fold():
    """Fold along a crease line with specified side"""
    data = request.json
    edge = tuple(data['edge'])
    side = int(data['side'])  # 1 or -1
    
    print(f"\n=== FOLD REQUEST ===")
    print(f"Edge to fold: {edge}")
    print(f"Side: {side}")
    print(f"Current faces before fold: {origami.faces}")
    print(f"Current orientations before fold: {origami.faces_orientations}")
    print(f"Current layers before fold: {origami.layers}")
    
    # Convert side parameter to vertex parameter for fold_on_crease
    # Find a vertex on the specified side of the line
    from utils import get_line_equasion, point_side_to_line
    v1_pos = origami.vertices[edge[0]]
    v2_pos = origami.vertices[edge[1]]
    line = get_line_equasion(v1_pos, v2_pos)
    
    # Find a vertex on the specified side
    vertex_to_fold = None
    for vid, vertex_pos in origami.vertices.items():
        vertex_side = point_side_to_line(vertex_pos, line)
        if vertex_side == side:
            vertex_to_fold = vid
            break
    
    if vertex_to_fold is None:
        return jsonify({
            'success': False,
            'error': f'No vertex found on side {side} of the crease line'
        })
    
    print(f"Using vertex {vertex_to_fold} for fold_on_crease (side {side})")
    
    save_state()
    try:
        origami.sss_fold_by_edge_and_vertex(edge, vertex_to_fold)
        print(f"Faces after fold: {origami.faces}")
        print(f"Orientations after fold: {origami.faces_orientations}")
        print(f"Layers after fold: {origami.layers}")
        return jsonify({
            'success': True,
            'state': get_origami_data()
        })
    except Exception as e:
        print(f"Fold error: {type(e).__name__}: {str(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        
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