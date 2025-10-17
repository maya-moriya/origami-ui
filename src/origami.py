"""Origami paper folding simulation module.

This module provides functionality to simulate paper folding operations
including creasing, splitting, and complex fold operations.
"""

import logging
import math
import sys
from typing import Dict, List, Set, Tuple, Optional, Union

import numpy as np
import matplotlib.pyplot as plt

from src.utils import *
from src.origami_base import OrigamiBase
from src.origami_visualizer import OrigamiVisualizer

# Configuration constants
DEBUG_MODE = True

# Logging configuration
logging.basicConfig(
    format='%(levelname)s | %(funcName)s | %(message)s',
    stream=sys.stdout 
)

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.WARNING)

class Origami(OrigamiBase):
    """Main origami class for paper folding simulation.
    
    Provides functionality for creating, folding, and manipulating
    virtual origami paper with support for complex folding operations.
    """
    
    def __init__(self, size: float = 10.0):
        """Initialize a square paper.
        
        Args:
            size: Side length of the square paper.
        """
        super().__init__()
        self.size = size
        self.initial_setup = {
            1: np.array([0.0, size]),
            2: np.array([size, size]),
            3: np.array([size, 0.0]),
            4: np.array([0.0, 0.0]),
        }
        self.vertices = self.initial_setup.copy()
        self.faces = {
            1: [1, 2, 3, 4]
        }
        self.faces_orientations = {
            1: 0,
        }
        self.layers = {
            1: [1],
        }
        self.edges_splits = []
        self.actions = []
        self.visualizer = OrigamiVisualizer(self)

    # ===== BASIC OPERATIONS =====
    def flip(self) -> None:
        """Flip the entire origami paper over."""
        max_layer = max(self.layers.keys())
        new_layers = {}
        for lid, faces in self.layers.items():
            new_lid = max_layer - lid + 1
            new_layers[new_lid] = faces
            for fid in faces:
                self.faces_orientations[fid] = 1 - self.faces_orientations[fid]
        self.layers = new_layers
        for vid in self.vertices.keys():
            self.vertices[vid][0] = self.size - self.vertices[vid][0]

    # Note: _check_vid_exists, _check_fid_exists, and vid_in_face are now in base class

    
    # Note: Edge and face utility methods are now in OrigamiBase
    
    # ===== ACTION TRACKING METHODS =====
    def _note_split(self, edge: Tuple[int, int], ratio: float) -> None:
        """Record a split action for history tracking."""
        self.actions.append({"action": "split", "edge": edge, "ratio": ratio})

    def _note_fold(self, edge: Tuple[int, int], side: int) -> None:
        """Record a fold action for history tracking."""
        self.actions.append({"action": "fold", "edge": edge, "side": side})
    
    # ===== GEOMETRIC OPERATIONS =====
    def _get_vertex_side_to_line(self, v_id: int, line) -> int:
        """Get the side of a vertex relative to a line."""
        self._check_vid_exists(v_id)
        return point_side_to_line(self.vertices[v_id], line)
    
    def _get_line_equation(self, v1_id: int, v2_id: int):
        """Get line equation from two vertices."""
        p1, p2 = self.vertices[v1_id], self.vertices[v2_id]
        return get_line_equasion(p1, p2)

    def _reflect_vertex(self, v_id: int, line_eq) -> None:
        """Reflect a vertex across a line."""
        self._check_vid_exists(v_id)
        self.vertices[v_id] = reflect_point(self.vertices[v_id], line_eq)

    def _get_edge_line_intersection(self, line, edge):
        """Get intersection ratio of line with edge."""
        v1_id, v2_id = edge
        p1, p2 = self.vertices[v1_id], self.vertices[v2_id]
        return segment_line_intersection(line, (p1, p2))
    
    # ===== FACE FINDING AND MANIPULATION =====
    def _find_face_containing_one_side_of_the_edge_and_vertex(self, edge, vertex):
        edge = tuple(sorted(edge))
        v1_id, v2_id = edge
        for fid, face in self.faces.items():
            if (v1_id in face and vertex in face) or (v2_id in face and vertex in face):
                return fid
        return None
    
    def _find_face_crossed_by_edge_and_containing_vertex(self, edge, vertex):
        edge = tuple(sorted(edge))
        v1_id, v2_id = edge
        face_with_edge_and_vertex = []
        for fid, face in self.faces.items():
            if v1_id in face and v2_id in face and vertex in face:
                if not self._edge_in_face(fid, edge):
                    face_with_edge_and_vertex.append(fid)
        if len(face_with_edge_and_vertex) > 1:
            raise ValueError("Multiple faces contain both the edge and the vertex.")
        if face_with_edge_and_vertex == []:
            return None
        return face_with_edge_and_vertex[0]

    def _split_face(self, fid, v1_id, v2_id):
        
        face = self.faces[fid]

        self._check_fid_exists(fid)
        self._check_vid_exists(v1_id)
        self._check_vid_exists(v2_id)
        if v1_id not in face or v2_id not in face:
            raise ValueError("One or both vertex IDs do not exist in face.")
        
        fisrt_index_in_face, sec_index_in_face = sorted([face.index(v1_id), face.index(v2_id)])
        face_1 = face[fisrt_index_in_face:sec_index_in_face + 1]
        face_2 = face[sec_index_in_face:] + face[:fisrt_index_in_face + 1]

        return face_1, face_2

    # Note: Layer operations moved to organized section above
    
    # ===== LAYER OPERATIONS =====
    def _add_face_to_layer(self, fid: int, lid: int) -> None:
        """Add a face to a specific layer."""
        self._check_fid_exists(fid)
        if lid in self.layers.keys():
            self.layers[lid].append(fid)
        else:
            self.layers[lid] = [fid]
    
    def _find_layer_for_new_face(self, old_lid: int, new_face: List[int], debug: bool = False) -> int:
        """Find an appropriate layer for a new face to avoid overlaps."""
        logging.debug(f"Finding layer for new face {new_face} from old layer {old_lid}")
        lid = old_lid
        while True:
            if lid not in self.layers.keys():
                logging.debug(f"Layer {lid} is not in layers, returning it.")
                return lid
            if self._face_overlap_with_layer(lid, new_face):
                logging.debug(f"Layer {lid} overlaps with new face {new_face}, checking next layer.")
                lid += 1
            else:
                logging.debug(f"Layer {lid} does not overlap with new face {new_face}, returning it.")
                return lid
    
    def _all_vertices_on_one_side_of_line(self, face, vids_side_map):
        sign = None
        for vid in face:
            vid_sign = vids_side_map[vid]
            if vid_sign != 0:
                if sign is None:
                    sign = vids_side_map[vid]
                elif sign != vids_side_map[vid]:
                    return None
        logging.debug(f"All vertices of face {face} are on one side of the line: {sign}: {[vids_side_map[vid] for vid in face]}")
        return sign

    def _complete_faces(self, face1_set, face2_set):
        shared_vids = face1_set & face2_set
        if len(shared_vids) in {1, 2}:
            shared_line = self._get_line_equation(*tuple(shared_vids))
            vids_side_map = self._vids_side_map(shared_line)
            face1_sign = self._all_vertices_on_one_side_of_line(face1_set, vids_side_map)
            face2_sign = self._all_vertices_on_one_side_of_line(face2_set, vids_side_map)
            logging.debug(f"Faces {face1_set} and {face2_set} share an edge {shared_vids}. Face1 sign: {face1_sign}, Face2 sign: {face2_sign}")
            if face1_sign is not None and face2_sign is not None and face1_sign != face2_sign:
                return True
        return False

    def _face_overlap_with_layer(self, layer: int, new_face: List[int]) -> bool:
        """Check if a new face overlaps with any face in a layer."""
        logging.debug(f"Checking overlap of new face {new_face} with layer {layer} with faces {self.layers.get(layer, [])}")

        layer_faces = self.layers.get(layer, [])
        logging.debug(f"Faces in layer {layer_faces}: {[self.faces[fid] for fid in layer_faces]}")

        for fid in layer_faces:
            logging.debug(f"Checking face {fid} {self.faces[fid]} against new face {new_face}")

            face = self.faces[fid]
            face_set = set(face)
            new_face_set = set(new_face)

            if face_set == new_face_set:
                logging.debug(f"Face {fid} {self.faces[fid]} is identical to new face {new_face}")
                return True

            # # Check if faces share exactly 2 vertices (an edge)
            # if self._complete_faces(face_set, new_face_set):
            #     logging.debug(f"Face {fid} {self.faces[fid]} shares an edge with new face {new_face}, skipping overlap check.")
            #     continue
            face1 = [tuple(self.vertices[vid]) for vid in self.faces[fid]]
            face2 = [tuple(self.vertices[vid]) for vid in new_face]
            if do_faces_overlap(face1, face2):
                logging.debug(f"Face {fid} {self.faces[fid]} overlaps with new face {new_face}")
                return True
            
            else:
                logging.debug(f"Face {fid} {self.faces[fid]} does not overlap with new face {new_face}")
        return False
    
    # ===== VERTEX OPERATIONS =====
    def _vertices_sides_list(self, vertices: List[int], line) -> Dict[int, Set[int]]:
        """Get side mapping of vertices relative to a line."""
        side_map = {1: set(), -1: set(), 0: set()}
        for vid in vertices:
            side = self._get_vertex_side_to_line(vid, line)
            side_map[side].add(vid)
        return side_map
    
    def _get_vertices_of_face_side_to_line(self, fid, line):
        self._check_fid_exists(fid)
        face = self.faces[fid]
        return self._vertices_sides_list(face, line)

    def _get_min_layer_in_faces(self, faces: List[int]) -> int:
        """Get the minimum layer number among given faces."""
        face_layer_map = self._face_layer_map()
        return min([face_layer_map[fid] for fid in faces])
    
    def _sort_faces_by_layers(self, faces: List[int], reverse: bool = True) -> List[int]:
        """Sort faces by their layer order."""
        fids = list(faces)
        face_layer_map = self._face_layer_map()
        return sorted(fids, key=lambda fid: face_layer_map[fid], reverse=reverse), face_layer_map

    def _cut_face_along_line(self, crease_line, fid):      
        EPS = 1e-5  
        self._check_fid_exists(fid)

        face = self.faces[fid]
        logging.debug(f"Cutting face {fid} {face} along line {crease_line}")

        vertices = set()
        for i in range(len(face)):
            v1_id = face[i]
            v2_id = face[(i + 1) % len(face)]
            edge =  tuple(sorted((v1_id, v2_id)))
            v1_id, v2_id = edge
            current_edge_line = self._get_line_equation(v1_id, v2_id)
            if crease_line == current_edge_line:
                logging.debug(f"Line {crease_line} is identical to edge {edge}")
                vertices.add(v1_id)
                vertices.add(v2_id)
            else:
                intersection_ratio = self._get_edge_line_intersection(crease_line, edge)
                logging.debug(f"Edge {edge} intersection ratio with line {crease_line}: {intersection_ratio}")
                if intersection_ratio is None:
                    continue
                if abs(intersection_ratio - 0) < EPS:
                    vertices.add(v1_id)
                elif abs(intersection_ratio - 1) < EPS:
                    vertices.add(v2_id)
                elif intersection_ratio < 1 and intersection_ratio > 0:
                    new_vid = self.split_face_by_edge_and_ratio(edge, intersection_ratio)
                    vertices.add(new_vid)
    
        if len(vertices) == 1:
            intersection_vid = list(vertices)[0]
            if intersection_vid in face:
                return None
            else:
                raise ValueError("Only one intersection vertex found, but it is not in the face.")
        
        if len(vertices) == 0:
            return None
        
        if len(vertices) > 2:
            raise ValueError("More than two intersection vertices found, cannot define a single edge.")
        
        return tuple(sorted(vertices))

    # ===== FOLDING OPERATIONS =====

    def sss_get_highest_cutted_face(self, sorted_faces, line, side_to_fold):
        for fid in sorted_faces:
            logging.debug(f"Checking face {fid} {self.faces[fid]} for cutting by line {line}")
            edge = self._cut_face_along_line(line, fid)
            # if edge is not None and not self._edge_in_face(fid, edge):
            if edge is not None:
                vids_sides = self._side_vids_map_by_fid(fid, line)
                if len(vids_sides[side_to_fold]) > 0:
                    logging.debug(f" Cut edge for face {fid}: {edge}")
                    return fid, edge

    def sss_get_one_deep_chained_faces(self, initial_fid, fold_side, vertices_sides, vertices_faces):
        logging.debug(f"Finding one-deep chained faces for initial face {initial_fid} on side {fold_side}")

        face_vertices_on_side = {vid for vid in self.faces[initial_fid] if vertices_sides[vid] == fold_side}
        logging.debug(f"Face vertices on side {fold_side}: {face_vertices_on_side}")

        bunch_fids = {initial_fid}

        for vid in face_vertices_on_side:
            for fid in vertices_faces[vid]:
                logging.debug(f" Found face {fid} for vertex {vid} on side {fold_side}")
                if fid not in bunch_fids:
                    logging.debug(f" Adding face {fid} to bunch")
                    bunch_fids.add(fid)
        
        logging.debug(f"One-deep chained faces for {initial_fid}: {bunch_fids}")
        return bunch_fids

    def sss_get_chained_faces(self, vertices_sides, vertices_faces, initial_fid, fold_side, faces_already_checked):

        logging.debug(f"Finding chained faces for initial face {initial_fid} on side {fold_side}")

        bunch_fids = {initial_fid}
        last_iteration = None

        while last_iteration != bunch_fids:
            last_iteration = bunch_fids.copy()

            for fid in list(last_iteration):

                if fid not in faces_already_checked:
                    logging.debug(f" Expanding from face {fid}")
                    new_fids = self.sss_get_one_deep_chained_faces(fid, fold_side, vertices_sides, vertices_faces)
                    faces_already_checked.add(fid)

                    logging.debug(f" Found new faces: {new_fids}")
                    bunch_fids.update(new_fids)
        
        logging.debug(f"Final chained faces for {initial_fid}: {bunch_fids}")
        return bunch_fids
    
    def sss_update_intermidiate_faces(self, set_of_faces, faces_already_checked, vids_side_map, side_to_fold):
        min_layer = self._get_min_layer_in_faces(set_of_faces)
        logging.debug(f"Updating intermidiate faces from layer {min_layer} and above")

        for lid in self.layers.keys():
            if lid >= min_layer:
                for fid in self.layers[lid]:
                    vids_on_side_to_fold = [vid for vid in self.faces[fid] if vids_side_map[vid] == side_to_fold]
                    if len(vids_on_side_to_fold) > 0:
                        if fid not in faces_already_checked:
                            set_of_faces.add(fid)
                            faces_already_checked.add(fid)
        return set_of_faces
    
    def sss_get_bunch_from_top(self, initial_fid, vertices_sides, vertices_faces, side_to_fold):

        faces_already_checked = set()

        bunch_fids = {initial_fid}
        logging.debug(f"Starting bunch with initial face {initial_fid}")

        last_iteration = None

        while last_iteration != bunch_fids:
            last_iteration = bunch_fids.copy()

            for fid in list(last_iteration):

                logging.debug(f" Checking face {fid} in bunch")

                if fid not in faces_already_checked:
                    logging.debug(f" Expanding from face {fid}")

                    new_fids = self.sss_get_chained_faces(vertices_sides, vertices_faces, fid, side_to_fold, faces_already_checked)
                    logging.debug(f" Found new chained faces: {new_fids}")

                    bunch_fids.update(new_fids)
                    new_fids = self.sss_update_intermidiate_faces(bunch_fids, faces_already_checked, vertices_sides, side_to_fold)
                    logging.debug(f" Updated bunch with intermidiate faces: {new_fids}")

                    bunch_fids.update(new_fids)
                    faces_already_checked.add(fid)

        return bunch_fids

    def sss_get_first_vertex_which_is_not_edge(self, edge, face):
        for vid in face:
            if vid != edge[0] and vid != edge[1]:
                return vid
        return None

    def sss_face_vertices_to_reflect(self, folded_face, edge, line, vertices_sides, vertices_to_reflect):
        if edge is None:
            logging.debug(f" Reflecting all vertices of face {folded_face}")
        for vid in folded_face:
            if edge is None or (vid != edge[0] and vid != edge[1]):
                if vid not in vertices_to_reflect:
                    vertices_to_reflect.add(vid)

    def sss_find_layer(self, folded_face):
        new_lid = 1
        while self._face_overlap_with_layer(new_lid, folded_face):
            logging.debug(f"Folded face {folded_face} overlap in layer {new_lid}")
            new_lid += 1
        logging.debug(f"Layer for the new folded face: {new_lid}")
        return new_lid
    
    def sss_fold_bunch(self, line, side_to_fold, bunch, sorted_faces, face_layer_map, vertices_sides):
        # Split each face if needed
        vertices_to_reflect = set()

        folded_fids = []
        folded_faces = []
        folded_fids_orientations = []
        new_fid = max(self.faces.keys()) + 1
        for fid in sorted_faces:
            logging.debug(f"############################################ Processing face: {fid} ############################################")
            if fid in bunch:
                logging.debug(f"Face {fid} is in bunch")

                lid = face_layer_map[fid]
                logging.debug(f"Face {fid} is in layer {lid}")

                edge = self._cut_face_along_line(line, fid)
                logging.debug(f"Cut edge for face {fid}: {edge}")

                if edge is None or self._edge_in_face(fid, edge):
                    logging.debug(f"No split needed only reflection")
                    self.sss_face_vertices_to_reflect(self.faces[fid], edge, line, vertices_sides, vertices_to_reflect)
                    folded_fids.append(fid)
                    folded_faces.append(self.faces[fid])
                    folded_fids_orientations.append(1 - self.faces_orientations[fid])
                    logging.debug(f"Adding {fid} to folded fids : {folded_fids} {folded_faces} {folded_fids_orientations}")
                    self.layers[lid].remove(fid)
                    logging.debug(f"Removed face {fid} from layer {lid}")

                else:
                    logging.debug(f"Face {fid} needs to be split along edge {edge}")
                    edge_vid1, edge_vid2 = edge

                    face1, face2 = self._split_face(fid, edge_vid1, edge_vid2)
                    logging.debug(f"Split face {fid} into {face1} and {face2}")

                    vid = self.sss_get_first_vertex_which_is_not_edge(edge, face1)
                    folded_face, staying_face = (face1, face2) if vertices_sides[vid] == side_to_fold else (face2, face1)
                    logging.debug(f"Folding face: {folded_face}, Staying face: {staying_face}")

                    self.sss_face_vertices_to_reflect(folded_face, edge, line, vertices_sides, vertices_to_reflect)
                    
                    logging.debug(f"Replaces original fid {fid} with {staying_face} instead of {self.faces[fid]}")
                    self.faces[fid] = staying_face
                    
                    # Find new layer for the folded face
                    new_lid = self.sss_find_layer(folded_face)

                    folded_fids.append(new_fid)
                    folded_faces.append(folded_face)
                    folded_fids_orientations.append(1 - self.faces_orientations[fid])
                    logging.debug(f"Adding {new_fid} to folded fids : {folded_fids} {folded_faces} {folded_fids_orientations}")

                    logging.debug(f"Updated faces: {self.faces}")
                    logging.debug(f"Updated faces orientations: {self.faces_orientations}")
                    logging.debug(f"Updated layers: {self.layers}")

                    new_fid += 1

        logging.debug(f"------------------------------------------- Vertices to reflect: {vertices_to_reflect} -------------------------------------------")
        for vid in vertices_to_reflect:
            logging.debug(f"Reflecting vertex {vid} at position {self.vertices[vid]} across line {line}")
            self._reflect_vertex(vid, line)

        logging.debug(f"------------------------------------------- Faces to update: {folded_fids} -------------------------------------------")
        for fid, face, orientation in zip(folded_fids, folded_faces, folded_fids_orientations):
            logging.debug(f"Adding folded face {fid} {face} with orientation {orientation}")
            new_lid = self.sss_find_layer(face)
            logging.debug(f" New layer for folded face {fid}: {new_lid}")
            if fid not in self.faces:
                logging.debug(f" Adding new folded face {fid} to faces")
                self.faces[fid] = face
            logging.debug(f" Adding folded face {fid} to layer {new_lid}")
            self.layers[new_lid] = self.layers.get(new_lid, []) + [fid]
            logging.debug(f" Updated orientations: {orientation}")
            self.faces_orientations[fid] = orientation            

    def sss_fold_by_edge_and_vertex(self, edge, vertex_on_side_to_fold):
        logging.debug(f"Folding by edge {edge} and vertex {vertex_on_side_to_fold}")
        self._check_vid_exists(vertex_on_side_to_fold)
        line = self._get_line_equation(*edge)
        side_to_fold = self._get_vertex_side_to_line(vertex_on_side_to_fold, line)
        logging.debug(f"Vertex {vertex_on_side_to_fold} is on side {side_to_fold} of line.")
        self.sss_fold_highest_bunch(line, side_to_fold)

    def sss_fold_highest_bunch(self, line, side_to_fold=1):
        bunch, sorted_faces, face_layer_map, vertices_sides, vertices_faces = self.sss_collect_info_and_get_highest_bunch(line, side_to_fold)
        logging.debug(f"Bunch to fold: {bunch}")
        self.sss_fold_bunch(line, side_to_fold, bunch, sorted_faces, face_layer_map, vertices_sides)
        logging.debug(f"Finished folding bunch")

    def sss_collect_info_and_get_highest_bunch(self, line, side_to_fold=1):
        sorted_faces, face_layer_map, vertices_sides, vertices_faces = self.sss_collect_info_for_fold(line)
        return self.sss_get_highest_bunch(line, sorted_faces, face_layer_map, vertices_sides, vertices_faces, side_to_fold)
    
    def sss_get_highest_bunch(self, line, sorted_faces, face_layer_map, vertices_sides, vertices_faces, side_to_fold=1):

        initial_fid, edge = self.sss_get_highest_cutted_face(sorted_faces, line, side_to_fold)
        logging.debug(f"Initial face to fold: {initial_fid} with edge {edge}")
        
        # After cutting, update vertices_sides and vertices_faces to include any new vertices created
        vertices_sides = self._vids_side_map(line)
        vertices_faces = self._face_vertex_map()

        bunch = self.sss_get_bunch_from_top(initial_fid, vertices_sides, vertices_faces, side_to_fold)
        logging.debug(f"Bunch of faces to fold: {bunch}")

        return bunch, sorted_faces, face_layer_map, vertices_sides, vertices_faces
    
    def sss_collect_info_for_fold(self, line):
        # Collect all faces from the toppest layer
        sorted_faces, face_layer_map = self._sort_faces_by_layers(self.faces.keys())
        vertices_sides = self._vids_side_map(line)
        vertices_faces = self._face_vertex_map()

        logging.debug(f"faces: {self.faces}")
        logging.debug(f"Sorted faces by layers: {sorted_faces}")
        logging.debug(f"Face layer map: {face_layer_map}")
        logging.debug(f"Vertices sides map: {vertices_sides}")
        logging.debug(f"Vertices faces map: {vertices_faces}")

        return sorted_faces, face_layer_map, vertices_sides, vertices_faces




    def get_crease_faces(self, v1_id: int, v2_id: int, vertex_to_fold: Optional[int] = None) -> List[int]:
        """Get faces that contain the crease edge or are affected by it."""
        faces_with_both = []
        faces_with_one = []
        for fid, face in self.faces.items():
            if v1_id in face and v2_id in face:
                faces_with_both.append(fid)
            elif v1_id in face or v2_id in face:
                faces_with_one.append(fid)
        if len(faces_with_both) > 0:
            if vertex_to_fold is not None:
                faces_with_all = [fid for fid in faces_with_both if vertex_to_fold in self.faces[fid]]
                if len(faces_with_all) > 0:
                    logging.debug(f"Found faces with both crease vertices and the folding vertex: {[self.faces[fid] for fid in faces_with_all]}")
                    return faces_with_all
            logging.debug(f"Found faces with both crease vertices: {[self.faces[fid] for fid in faces_with_both]}")
            return faces_with_both
        if len(faces_with_one) > 0:
            if vertex_to_fold is not None:
                faces_with_all = [fid for fid in faces_with_one if vertex_to_fold in self.faces[fid]]
                if len(faces_with_all) > 0:
                    logging.debug(f"Found faces with one crease vertex and the folding vertex: {[self.faces[fid] for fid in faces_with_all]}")
                    return faces_with_all
            logging.debug(f"Found faces with one crease vertex: {[self.faces[fid] for fid in faces_with_one]}")
            return faces_with_one
        
        raise ValueError("No faces found containing either vertex of the edge.")
    
    def _vids_side_map(self, line):
        vids_side_map = {}
        for vid in self.vertices.keys():
            side = self._get_vertex_side_to_line(vid, line)
            vids_side_map[vid] = side
        logging.debug(f"Vertices side map: {vids_side_map}")
        return vids_side_map
    
    def _side_vids_map_by_fid(self, fid, line):
        side_vids_map = {1: [], -1: [], 0: []}
        for vid in self.faces[fid]:
            side = self._get_vertex_side_to_line(vid, line)
            side_vids_map[side].append(vid)
        logging.debug(f"Sides vertices map of face {fid}: {side_vids_map}")
        return side_vids_map

    def update_vertices_side_map(self, line, existing_map):
        new_vertices = set(self.vertices.keys()) - set(existing_map.keys())
        for vid in new_vertices:
            side = self._get_vertex_side_to_line(vid, line)
            existing_map[vid] = side
            logging.debug(f" Added new vertex to side map. Vertices side map: {existing_map}")
        return existing_map

    def _prepare_face_split_info(self, fid: int, line, vids_side_map: Dict[int, int]) -> Dict:
        """Prepare split information for a single face."""
        face = self.faces[fid]
        fids_layer_map = self._face_layer_map()
        
        face_info = {
            "crease": None,
            "split": False,
            "lid": fids_layer_map[fid],
            "vertices": {1: set(), 0: set(), -1: set()},
            "faces": {1: None, -1: None}
        }
        
        # Assign vertices to side groups
        for vid in face:
            side = vids_side_map[vid]
            face_info["vertices"][side].add(vid)
        
        return face_info
    
    def _determine_face_crease(self, fid: int, face_info: Dict, line, v1_id: int, v2_id: int) -> None:
        """Determine the crease for a face."""
        face = self.faces[fid]
        if v1_id in face and v2_id in face:
            face_info['crease'] = tuple(sorted((v1_id, v2_id)))
        else:
            face_info['crease'] = self._cut_face_along_line(line, fid)
    
    def _handle_face_splitting(self, fid: int, face_info: Dict) -> None:
        """Handle splitting of faces that cross the fold line."""
        if len(face_info["vertices"][1]) > 0 and len(face_info["vertices"][-1]) > 0:
            face_info['split'] = True
            crease_vid1, crease_vid2 = face_info['crease']
            face1, face2 = self._split_face(fid, crease_vid1, crease_vid2)
            
            if list(face_info["vertices"][1])[0] in face1:
                face_info["faces"][1] = face1
                face_info["faces"][-1] = face2
            else:
                face_info["faces"][1] = face2
                face_info["faces"][-1] = face1
            
            logging.debug(f"    Face {fid} will be split into two faces: {face1} and {face2}")
        
        elif len(face_info["vertices"][1]) > 0:
            face_info["faces"][1] = self.faces[fid]
            logging.debug(f"    All vertices of face {fid} are on the positive side of the line.")
        
        elif len(face_info["vertices"][-1]) > 0:
            face_info["faces"][-1] = self.faces[fid]
            logging.debug(f"    All vertices of face {fid} are on the negative side of the line.")
        else:
            raise ValueError("Face has no vertices on either side of the line.")

    def fold_preparations(self, edge: Tuple[int, int]) -> Tuple:
        """Prepare all necessary information for folding along an edge.
        
        Args:
            edge: Tuple of vertex IDs forming the fold line
            
        Returns:
            Tuple of (line equation, faces split information)
        """
        logging.debug(f" Preparing to fold along edge {edge}")
        v1_id, v2_id = edge
        line = self._get_line_equation(v1_id, v2_id)
        vids_side_map = self._vids_side_map(line)
        faces_split_info = {}
        
        for fid, face in self.faces.items():
            # Update side map if new vertices were added
            if len(self.vertices) > len(vids_side_map):
                vids_side_map = self.update_vertices_side_map(line, vids_side_map)
            
            logging.debug(f" Preparing face {fid}: {face}")
            
            # Prepare basic face information
            faces_split_info[fid] = self._prepare_face_split_info(fid, line, vids_side_map)
            
            # Determine the crease for this face
            self._determine_face_crease(fid, faces_split_info[fid], line, v1_id, v2_id)
            
            # Handle face splitting logic
            self._handle_face_splitting(fid, faces_split_info[fid])
        
        return line, faces_split_info

    def _get_faces_to_fold(self, faces_split_info, pos_to_fold, min_layer):
        logging.debug(f" Getting faces to fold on the {pos_to_fold} side above layer {min_layer}")
        #return {fid for fid, info in faces_split_info.items() if (info['lid'] >= min_layer and info['faces'][pos_to_fold] is not None)}
        faces_to_fold = set()
        for fid, info in faces_split_info.items():
            if info['lid'] >= min_layer and info['faces'][pos_to_fold] is not None:
                logging.debug(f"  Adding face {fid} to fold list, since layer {info['lid']} >= {min_layer} and has vertices on the {pos_to_fold} side: {info['faces'][pos_to_fold]}")
                faces_to_fold.add(fid)
        vertices_faces_map = self._face_vertex_map()
        last_iteration = None
        while last_iteration != faces_to_fold:
            last_iteration = faces_to_fold.copy()
            for fid in list(last_iteration):
                logging.debug(f" Checking adjacent faces for face {fid}: vertices on the folding side: {faces_split_info[fid]['vertices'][pos_to_fold]}")
                for vid in list(faces_split_info[fid]['vertices'][pos_to_fold]):
                    for adjacent_fid in vertices_faces_map[vid]:
                        if fid != adjacent_fid and adjacent_fid not in faces_to_fold:
                            logging.debug(f" {adjacent_fid} is adjacent to {fid} through vertex {vid}")
                            faces_to_fold.add(adjacent_fid)
        return faces_to_fold
    
    def fold_on_crease(self, edge, vertex_to_fold):
        line = self._get_line_equation(*edge)
        pos_to_fold = self._get_vertex_side_to_line(vertex_to_fold, line)
        self.fold_on_crease_by_side(edge, pos_to_fold, vertex_to_fold)

    def get_two_fold_options(self, edge):
        line, faces_split_info = self.fold_preparations(edge)
        v1_id, v2_id = edge
        initial_faces_to_fold = self.get_crease_faces(v1_id, v2_id)
        min_layer = self._get_min_layer_in_faces(initial_faces_to_fold)
        faces_to_fold_positive = self._get_faces_to_fold(faces_split_info, 1, min_layer)
        faces_to_fold_negative = self._get_faces_to_fold(faces_split_info, -1, min_layer)
        return line, faces_split_info, faces_to_fold_positive, faces_to_fold_negative

    def _update_faces_after_split(self, faces_to_fold, pos_to_fold, faces_split_info):
        vids_to_reflect = set()
        logging.debug(f"Starting to fold faces...")
        for fid in faces_to_fold:
            logging.debug(f" Folding face {fid} in layer {faces_split_info[fid]['lid']} with split={faces_split_info[fid]['split']} and vertices sides: {faces_split_info[fid]['faces']}")
            info = faces_split_info[fid]
            if info['split']:
                new_fid = max(self.faces.keys()) + 1
                pos_to_stay = (-1) * pos_to_fold
                staying_face, folding_face = info['faces'][pos_to_stay], info['faces'][pos_to_fold]
                logging.debug(f"  Face {fid} is split. Staying face: {staying_face}, folding face: {folding_face} into new face ID {new_fid}")
                self.faces[fid] = staying_face
                self.faces[new_fid] = folding_face
                logging.debug(f"  Updated faces: {self.faces}")
                self.faces_orientations[new_fid] = 1 - self.faces_orientations[fid]
                faces_split_info[fid]['new_fid'] = new_fid
            else:
                if info['faces'][pos_to_fold] is not None:
                    self.faces_orientations[fid] = 1 - self.faces_orientations[fid]
            for vid in info['vertices'][pos_to_fold]:
                    vids_to_reflect.add(vid)
        return vids_to_reflect, faces_split_info

    def _arrange_faces_to_fold(self, edge, faces_split_info, pos_to_fold, vertex_to_fold):

        v1_id, v2_id = edge
        initial_faces_to_fold = self.get_crease_faces(v1_id, v2_id, vertex_to_fold)

        logging.debug(f"Initial faces to fold: {initial_faces_to_fold}")
        min_layer = self._get_min_layer_in_faces(initial_faces_to_fold)

        logging.debug(f"Minimal layer among these faces: {min_layer}")
        faces_to_fold = self._get_faces_to_fold(faces_split_info, pos_to_fold, min_layer)

        logging.debug(f"Faces to fold before sorting, above layer {min_layer} and in {pos_to_fold} side: {faces_to_fold}")
        faces_to_fold = sorted(faces_to_fold, key=lambda fid: faces_split_info[fid]['lid'], reverse=True)

        return faces_to_fold, min_layer
    
    def _update_layers_after_split(self, faces_to_fold, min_layer, faces_split_info):
        new_layer = None
        for fid in faces_to_fold:
            info = faces_split_info[fid]
            old_layer = info['lid']
            if info['split']:
                new_fid = faces_split_info[fid]['new_fid']
            else:
                new_fid = fid
                self.layers[old_layer].remove(fid)
            if new_layer is None:
                # For split faces, start from min_layer
                # For faces that are folded entirely, always start from layer 1 to find overlaps correctly
                search_start = min_layer if info['split'] else 1
                new_layer = self._find_layer_for_new_face(search_start, self.faces[new_fid])
            else:
                new_layer = self._find_layer_for_new_face(new_layer, self.faces[new_fid])
            if info['split']:
                logging.debug(f"  Face {fid} {self.faces[fid]} is folded. New face ID {new_fid} will be in layer {new_layer}.")
            else:
                logging.debug(f"  Face {fid} {self.faces[fid]} is folded entirely. Old layer {old_layer}, Moving to layer {new_layer}. self.layers: {self.layers}")

            if new_layer in self.layers.keys():
                self.layers[new_layer].append(new_fid)
            else:
                self.layers[new_layer] = [new_fid]
            logging.debug(f" Moved face {new_fid} to layer {new_layer}. self.layers: {self.layers}")


    def fold_on_crease_by_side(self, edge, pos_to_fold, vertex_to_fold=None):
        if pos_to_fold not in [1, -1]:
            raise ValueError("pos_to_fold must be 1 (positive side) or -1 (negative side).")
        
        self._note_fold(edge, pos_to_fold)
        line, faces_split_info = self.fold_preparations(edge)
        
        logging.debug(f"############# Folding along edge {edge} to the {pos_to_fold} side of the line #############")
        
        faces_to_fold, min_layer = self._arrange_faces_to_fold(edge, faces_split_info, pos_to_fold, vertex_to_fold)

        logging.debug(f"Faces to fold sorted by layers: {faces_to_fold}")
        vids_to_reflect, faces_split_info = self._update_faces_after_split(faces_to_fold, pos_to_fold, faces_split_info)

        logging.debug(f"Updating layers after splitting... current layers: {[self.layers]} (faces: {self.faces})")
        self._update_layers_after_split(faces_to_fold, min_layer, faces_split_info)

        logging.debug(f"Vertices to reflect: {vids_to_reflect}")
        for vid in vids_to_reflect:
            self._reflect_vertex(vid, line)

    
    # ===== EDGE SPLITTING OPERATIONS =====
    def split_face_by_edge_and_ratio(self, edge: Tuple[int, int], ratio_list: Union[float, List[float]] = 0.5) -> Union[int, List[int]]:
        """Split an edge at given ratio(s) and update affected faces.
        
        Args:
            edge: Tuple of vertex IDs forming the edge
            ratio_list: Single ratio or list of ratios to split at
            
        Returns:
            New vertex ID(s) created at split point(s)
        """
        self._note_split(edge, ratio_list)

        single_cut = not isinstance(ratio_list, list)
        if single_cut:
            ratio_list = [ratio_list]

        for ratio in ratio_list:
            if ratio <= 0 or ratio >= 1:
                raise ValueError("Ratio must be between 0 and 1.")
        
        v1_id, v2_id = edge
        if v1_id > v2_id:
            for i in range(len(ratio_list)):
                ratio_list[i] = 1 - ratio_list[i]
        edge = tuple(sorted(edge))
        v1_id, v2_id = edge
        
        v1, v2 = self.vertices[v1_id], self.vertices[v2_id]
        new_ids = []
        ratio_list = sorted(ratio_list)
        for ratio in ratio_list:
            pos = (1 - ratio) * v1 + ratio * v2
            new_id = max(self.vertices.keys()) + 1
            new_ids.append(new_id)
            self.vertices[new_id] = pos

        for fid, face in self.faces.items():
            if v1_id in face and v2_id in face:
                if edge == tuple(sorted((face[0], face[-1]))):
                    for new_id in new_ids:
                        face.append(new_id)
                else:
                    first_index_in_face = min(face.index(v1_id), face.index(v2_id))
                    face = face[:first_index_in_face + 1] + new_ids + face[first_index_in_face + 1:]
                self.faces[fid] = face

        for i, id in enumerate(new_ids):
            self.edges_splits.append((id, edge, ratio_list[i]))

        if single_cut:
            return new_ids[0]
        
        return new_ids
    
    # Note: _get_all_edges is now in OrigamiBase
    
    def _find_face_with_3_vids(self, vid1, vid2, vid3):
        for fid, face in self.faces.items():
            if vid1 in face and vid2 in face and vid3 in face:
                return fid
        return None
    
    def _find_face_with_2_points(self, p1, p2):
        for fid, face in self.faces.items():
            if p1 in face and p2 in face:
                return fid
        return None
    
    def split_pocket_face(self, pivot_vid, shifted_vid, third_vid):
        face = self._find_face_with_3_vids(pivot_vid, shifted_vid, third_vid)
        if not self._edge_in_face(face, (pivot_vid, shifted_vid)):
            raise ValueError("The pivot vertex and shifted vertex do not form an edge in the face.")
        if not self._edge_in_face(face, (shifted_vid, third_vid)):
            raise ValueError("The shifted vertex and third vertex do not form an edge in the face.")
        t = find_angle_bisector_ratio(self.vertices[shifted_vid], self.vertices[pivot_vid], self.vertices[third_vid])
        new_vid = self.split_face_by_edge_and_ratio((shifted_vid, third_vid), t)
        return face, new_vid
    
    def get_pocket_staying_face(self, face, edge_vid1, edge_vid2, excluded_vid):
        face1, face2 = self._split_face(face, edge_vid1, edge_vid2)
        if excluded_vid in face1:
            return face2
        else:
            return face1
        
    def _remove_vid(self, v_id: int) -> int:
        """Remove a vertex by replacing it with the highest-ID vertex.
        
        Args:
            v_id: Vertex ID to remove
            
        Returns:
            The ID of the vertex that was moved to replace the removed one
        """
        if v_id not in self.vertices:
            raise ValueError(f"Cannot remove vertex {v_id}: does not exist")
        
        max_vid = max(self.vertices.keys())
        if v_id == max_vid:
            # If we're removing the highest ID vertex, just delete it
            del self.vertices[v_id]
            return v_id
        
        # Replace the vertex to remove with the highest ID vertex
        self.vertices[v_id] = self.vertices[max_vid]
        del self.vertices[max_vid]
        
        # Update all face references
        for fid, face in self.faces.items():
            if max_vid in face:
                face[face.index(max_vid)] = v_id
        
        return max_vid

    # ===== POCKET OPERATIONS =====
    def pocket(self, pivot_vid: int, shifted_vid: int, upper_vid: int, lower_vid: int) -> None:
        """Create a pocket fold using four vertices.
        
        Args:
            pivot_vid: Vertex ID of the pivot point
            shifted_vid: Vertex ID that will be shifted/removed
            upper_vid: Vertex ID of the upper boundary
            lower_vid: Vertex ID of the lower boundary
        """
        self._check_vid_exists(pivot_vid)
        self._check_vid_exists(shifted_vid)
        self._check_vid_exists(upper_vid)
        self._check_vid_exists(lower_vid)
        upper_face, upper_new_vid = self.split_pocket_face(pivot_vid, shifted_vid, upper_vid)
        lower_face, lower_new_vid = self.split_pocket_face(pivot_vid, shifted_vid, lower_vid)

        given_crease_line = self._get_line_equation(pivot_vid, upper_vid)
        staying_upper_face = self.get_pocket_staying_face(upper_face, pivot_vid, upper_vid, shifted_vid)
        staying_lower_face = self.get_pocket_staying_face(lower_face, pivot_vid, lower_new_vid, shifted_vid)
        self.faces[upper_face] = staying_upper_face
        self.faces[lower_face] = staying_lower_face

        # Create the flipped face
        fliped_face = max(self.faces.keys()) + 1
        self.faces[fliped_face] = [pivot_vid, upper_vid, upper_new_vid]
        self.faces_orientations[fliped_face] = 1 - self.faces_orientations[upper_face]
        self._reflect_vertex(upper_new_vid, given_crease_line)

        # Create the extended face
        extended_face = max(self.faces.keys()) + 1
        self.faces[extended_face] = [pivot_vid, upper_new_vid, lower_new_vid]
        self.faces_orientations[extended_face] = self.faces_orientations[upper_face]
        
        # Remove the shifted_point
        self._remove_vid(shifted_vid)
        
        # Update layers
        upper_layer = self._face_layer_map()[upper_face]
        third_layer = self._find_layer_for_new_face(upper_layer, self.faces[fliped_face])
        self._add_face_to_layer(fliped_face, third_layer)
        forth_layer = self._find_layer_for_new_face(third_layer, self.faces[extended_face])
        self._add_face_to_layer(extended_face, forth_layer)
        

    # ===== VISUALIZATION METHODS =====
    def plot_layers(self) -> None:
        """Plot each layer of the origami separately."""
        self.visualizer.plot_layers()


    def plot(self, show_vertices_indices: bool = True, show_all_edges: bool = True, 
             print_layers: bool = False) -> None:
        """Plot the current state of the origami paper with filled faces."""
        self.visualizer.plot(show_vertices_indices, show_all_edges, print_layers)
