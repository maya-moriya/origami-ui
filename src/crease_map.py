import numpy as np
import matplotlib.pyplot as plt

def get_edges(paper):
    edges = set()
    for fid, face in paper.faces.items():
        for i in range(len(face)):
            edge = tuple(sorted((face[i], face[(i + 1) % len(face)])))
            edges.add(edge)
    return edges

def get_edges_lengths(paper):
    edges = get_edges(paper)
    edges_lengths = {}
    for edge in edges:
        v1, v2 = edge
        p1 = paper.vertices[v1]
        p2 = paper.vertices[v2]
        length = np.linalg.norm(np.array(p1) - np.array(p2))
        edges_lengths[edge] = length
    return edges_lengths

def get_edges_faces(paper):
    faces_layers = get_faces_layers(paper)
    edges_faces = {}
    for fid, face in paper.faces.items():
        for i in range(len(face)):
            edge = tuple(sorted((face[i], face[(i + 1) % len(face)])))
            if edge not in edges_faces:
                edges_faces[edge] = []
            edges_faces[edge].append(fid)
    for edge in edges_faces.keys():
        faces_list = edges_faces[edge]
        edges_faces[edge] = sorted(faces_list, key=lambda fid: faces_layers[fid])
    return edges_faces

def get_faces_layers(paper):
    faces_layers = {}
    for lid, layer in paper.layers.items():
        for fid in layer:
            faces_layers[fid] = lid
    return faces_layers

def get_vallies_and_mountains(paper):
    edges_faces = get_edges_faces(paper)
    vallies_and_mountains = {}
    for edge in edges_faces.keys():
        faces = edges_faces[edge]
        if len(faces) == 2:
            orientation1 = paper.faces_orientations[faces[0]]
            orientation2 = paper.faces_orientations[faces[1]]
            if orientation1 == 0 and orientation2 == 1:
                vallies_and_mountains[edge] = 'V'
            elif orientation1 == 1 and orientation2 == 0:
                vallies_and_mountains[edge] = 'M'
        else:
            vallies_and_mountains[edge] = 'B'
    return vallies_and_mountains
            
def recover_vertices_positions(paper, size=10.0):
    # vertices = paper.initial_setup.copy()
    vertices = {
            1: np.array([0.0, size]),
            2: np.array([size, size]),
            3: np.array([size, 0.0]),
            4: np.array([0.0, 0.0]),
        }
    for vid, edge, ratio in paper.edges_splits:
        v1, v2 = edge
        p1 = vertices[v1]
        p2 = vertices[v2]
        new_pos = (1 - ratio) * p1 + ratio * p2
        vertices[vid] = new_pos
    return vertices

def plot_crease_map(paper):

    vallies_and_mountains = get_vallies_and_mountains(paper)
    vertices = recover_vertices_positions(paper)
    fig, ax = plt.subplots(figsize=(4, 4))


    for edge, fold_type in vallies_and_mountains.items():
        color = 'black'
        if fold_type == 'V':
            color = 'red'
        elif fold_type == 'M':
            color = 'blue'
        p1, p2 = vertices[edge[0]], vertices[edge[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=1.5, zorder=7)

    plt.show()




