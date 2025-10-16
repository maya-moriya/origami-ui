"""Visualization module for origami plotting operations."""

import math
from typing import Dict, List, Set, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np


class OrigamiVisualizer:
    """Handles visualization and plotting of origami structures."""
    
    def __init__(self, origami_instance):
        self.origami = origami_instance
    
    def plot_layers(self) -> None:
        """Plot each layer of the origami separately."""
        cols = 3
        rows = math.ceil(len(self.origami.layers) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
        axes = axes.flatten()
        
        for i, layer in enumerate(reversed(self.origami.layers.values())):
            for fid in layer:
                face = self.origami.faces[fid]
                coords = [self.origami.vertices[vid] for vid in face]
                x_coords = [p[0] for p in coords]
                y_coords = [p[1] for p in coords]
                orientation = self.origami.faces_orientations.get(fid, 0)
                color = 'white' if orientation == 0 else 'lightblue'
                
                axes[i].fill(x_coords, y_coords, color=color, alpha=0.5, 
                           edgecolor='none', zorder=5)
                
                len_face = len(face)
                for j in range(len_face):
                    p1 = self.origami.vertices[face[j]]
                    p2 = self.origami.vertices[face[(j + 1) % len_face]]
                    axes[i].plot([p1[0], p2[0]], [p1[1], p2[1]], 
                               'k-', linewidth=1.5, zorder=5)
            
            axes[i].set_xlim(-2, 12)
            axes[i].set_ylim(-2, 12)
        
        plt.show()

    def plot(self, show_vertices_indices: bool = True, show_all_edges: bool = True, 
             print_layers: bool = False) -> None:
        """Plot the current state of the origami paper with filled faces."""
        fig, ax = plt.subplots(figsize=(4, 4))
        
        if show_all_edges:
            all_edges = self.origami._get_all_edges()
            for edge in all_edges:
                p1, p2 = self.origami.vertices[edge[0]], self.origami.vertices[edge[1]]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                       'k-', linewidth=0.3, zorder=7)

        if show_vertices_indices:
            pos_dict = self._vertices_by_position()
            for pos, vids in pos_dict.items():
                for i, vid in enumerate(vids):
                    ax.text(pos[0], pos[1] - (i * 0.5) - 0.2, str(vid), 
                           ha='center', fontsize=9, va='bottom',
                           bbox=dict(boxstyle="circle,pad=0.05", fc="white", 
                                   alpha=1, ec="black", lw=0.5), zorder=10)

        faces_to_plot = self._sort_faces_by_layers(
            self.origami.faces.keys(), reverse=False)
        
        for face_id in faces_to_plot:
            face = self.origami.faces[face_id]
            len_face = len(face)

            coords = [self.origami.vertices[vid] for vid in face]
            orientation = self.origami.faces_orientations.get(face_id, 0)
            color = 'white' if orientation == 0 else 'lightblue'
            x_coords = [p[0] for p in coords]
            y_coords = [p[1] for p in coords]

            ax.fill(x_coords, y_coords, color=color, alpha=1, 
                   edgecolor='none', zorder=5)

            for i in range(len_face):
                p1 = self.origami.vertices[face[i]]
                p2 = self.origami.vertices[face[(i + 1) % len_face]]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                       'k-', linewidth=1.5, zorder=5)

        ax.set_xlim(-2, 12)
        ax.set_ylim(-2, 12)

        if print_layers:
            print(".... Faces by Layers ...")
            for layer in sorted(self.origami.layers.keys()):
                print(f"    Layer {layer}")
                for fid in self.origami.layers[layer]:
                    orientation = self.origami.faces_orientations[fid]
                    print(f"        Face {fid}: {self.origami.faces[fid]} "
                          f"Orientation: {orientation}")

        plt.show()
    
    def _vertices_by_position(self) -> Dict[Tuple[float, float], List[int]]:
        """Group vertices by their positions."""
        pos_dict = {}
        for vid, pos in self.origami.vertices.items():
            found_point_in_dict = False
            for point in pos_dict.keys():
                if np.allclose(point, pos, atol=1e-5):
                    pos_dict[point].append(vid)
                    found_point_in_dict = True
                    break
            if not found_point_in_dict:
                pos_dict[tuple(pos)] = [vid]
        return pos_dict
    
    def _sort_faces_by_layers(self, faces: List[int], reverse: bool = True) -> List[int]:
        """Sort faces by their layer order."""
        fids = list(faces)
        face_layer_map = self.origami._face_layer_map()
        return sorted(fids, key=lambda fid: face_layer_map[fid], reverse=reverse)