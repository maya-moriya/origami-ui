"""Base validation and utility methods for origami operations."""

from typing import Dict, List, Set, Tuple, Optional, Union
import numpy as np


class OrigamiBase:
    """Base class providing common validation and utility methods for origami operations."""
    
    def __init__(self):
        self.vertices: Dict[int, np.ndarray] = {}
        self.faces: Dict[int, List[int]] = {}
        self.faces_orientations: Dict[int, int] = {}
        self.layers: Dict[int, List[int]] = {}
        
    def _check_vid_exists(self, v_id: int) -> None:
        """Validate that a vertex ID exists."""
        if v_id not in self.vertices:
            raise ValueError(f"Vertex ID {v_id} does not exist.")

    def _check_fid_exists(self, fid: int) -> None:
        """Validate that a face ID exists."""
        if fid not in self.faces:
            raise ValueError(f"Face ID {fid} does not exist.")
    
    def vid_in_face(self, fid: int, v_id: int) -> bool:
        """Check if a vertex is in a face."""
        return v_id in self.faces[fid]

    def _edges_of_face(self, fid: int) -> Set[Tuple[int, int]]:
        """Get all edges of a face."""
        if fid not in self.faces:
            raise ValueError(f"Face ID {fid} does not exist.")
        
        face = self.faces[fid]
        edges = set()
        for i in range(len(face)):
            v1_id = face[i]
            v2_id = face[(i + 1) % len(face)]
            edge = tuple(sorted((v1_id, v2_id)))
            edges.add(edge)
        return edges
    
    def _edge_in_face(self, fid: int, edge: Tuple[int, int]) -> bool:
        """Check if an edge is in a face."""
        edge = tuple(sorted(edge))
        edges_of_face = self._edges_of_face(fid)
        return edge in edges_of_face

    def _faces_for_vertices(self, vertices_list: List[int]) -> List[int]:
        """Get all faces that contain any of the given vertices."""
        faces_list = set()
        for vid in vertices_list:
            for fid, face in self.faces.items():
                if vid in face:
                    faces_list.add(fid)
        return list(faces_list)
    
    def _face_vertex_map(self) -> Dict[int, Set[int]]:
        """Create a mapping from vertex ID to set of face IDs containing it."""
        face_vertex_map = {}
        for fid, face in self.faces.items():
            for vid in face:
                if vid in face_vertex_map:
                    face_vertex_map[vid].add(fid)
                else:
                    face_vertex_map[vid] = {fid}
        return face_vertex_map
    
    def _face_layer_map(self) -> Dict[int, int]:
        """Create a mapping from face ID to layer ID."""
        face_layer_map = {}
        for lid, layer in self.layers.items():
            for fid in layer:
                face_layer_map[fid] = lid
        return face_layer_map
    
    def _get_all_edges(self) -> Set[Tuple[int, int]]:
        """Get all edges in the origami."""
        edges = set()
        for fid in self.faces.keys():
            face_edges = self._edges_of_face(fid)
            edges.update(face_edges)
        return edges