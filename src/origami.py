import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import math
from src.utils import *
import logging

logging.basicConfig(
    level=logging.DEBUG, # Set the minimum level to display messages
    format='%(levelname)s | %(funcName)s | %(message)s'
)

DEBUG_MODE = True

if not DEBUG_MODE:
    logging.basicConfig(level=logging.WARNING)

class Origami:
    def __init__(self, size=10.0):
        """Initializes a square paper."""
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

    def flip(self):
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

    def _check_vid_exists(self, v_id):
        if v_id not in self.vertices:
            raise ValueError("Vertex ID does not exist.")

    def _check_fid_exists(self, fid):
        if fid not in self.faces:
            raise ValueError("Face ID does not exist.")
    
    def vid_in_face(self, fid, v_id):
        return v_id in self.faces[fid]

    def _note_split(self, edge, ratio):
        self.actions.append({"action": "split", "edge": edge, "ratio": ratio})

    def _note_fold(self, edge, side):
        self.actions.append({"action": "fold", "edge": edge, "side": side})
    
    def _edges_of_face(self, fid):
        if fid not in self.faces:
            raise ValueError("Face ID does not exist.")
        face = self.faces[fid]
        edges = set()
        for i in range(len(face)):
            v1_id = face[i]
            v2_id = face[(i + 1) % len(face)]
            edge = tuple(sorted((v1_id, v2_id)))
            edges.add(edge)
        return edges
    
    def _edge_in_face(self, fid, edge):
        edge = tuple(sorted(edge))
        edges_of_face = self._edges_of_face(fid)
        return edge in edges_of_face

    def _faces_for_vertices(self, vertices_list):
        faces_list = set()
        for vid in vertices_list:
            for fid, face in self.faces.items():
                if vid in face:
                    faces_list.add(fid)
        return list(faces_list)
    
    def _face_vertex_map(self):
        face_vertex_map = {}
        for fid, face in self.faces.items():
            for vid in face:
                if vid in face_vertex_map:
                    face_vertex_map[vid].add(fid)
                else:
                    face_vertex_map[vid] = {fid}
        return face_vertex_map
    
    def _face_layer_map(self):
        face_layer_map = {}
        for lid, layer in self.layers.items():
            for fid in layer:
                face_layer_map[fid] = lid
        return face_layer_map
    
    def _get_vertex_position_to_line(self, v_id, line):
        self._check_vid_exists(v_id)
        return point_position_to_line(self.vertices[v_id], line)
        
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

    def _face_overlap_with_layer(self, layer, new_face):
        for fid in layer:
            if self.faces[fid] == new_face:
                return True
            face1 = [tuple(self.vertices[vid]) for vid in self.faces[fid]]
            face2 = [tuple(self.vertices[vid]) for vid in new_face]
            if do_faces_overlap(face1, face2):
                return True
        return False
    
    def _find_layer_for_new_face(self, old_lid, new_face, debug=False):
        logging.debug(f"Finding layer for new face {new_face} from old layer {old_lid}")
        lid = old_lid + 1
        while True:
            if lid not in self.layers.keys():
                logging.debug(f"Layer {lid} is not in layers, returning it.")
                return lid
            if self._face_overlap_with_layer(self.layers[lid], new_face):
                logging.debug(f"Layer {lid} overlaps with new face {new_face}, checking next layer.")
                lid += 1
            else:
                logging.debug(f"Layer {lid} does not overlap with new face {new_face}, returning it.")
                return lid
    
    def _add_face_to_layer(self, fid, lid):
        self._check_fid_exists(fid)
        if lid in self.layers.keys():
            self.layers[lid].append(fid)
        else:
            self.layers[lid] = [fid]
    
    def _vertices_positions_list(self, vertices, line):
        position_map = {1: set(), -1: set(), 0: set()}
        for vid in vertices:
            position = self._get_vertex_position_to_line(vid, line)
            position_map[position].add(vid)
        return position_map
    
    def _get_vertices_of_face_position_to_line(self, fid, line):
        self._check_fid_exists(fid)
        face = self.faces[fid]
        return self._vertices_positions_list(face, line)

    # def _get_faces_on_side_of_line(self, line, side):
    #     if side not in [1, -1]:
    #         raise ValueError("Side must be 1 (left) or -1 (right).")
    #     faces_on_side = set()
    #     for fid in self.faces.keys():
    #         position_map = self._get_vertices_of_face_position_to_line(fid, line)
    #         if len(position_map[side]) > 0:
    #             faces_on_side.add(fid)
    #     return faces_on_side
    
    def _get_min_layer_in_faces(self, faces):
        face_layer_map = self._face_layer_map()
        min_layer = min([face_layer_map[fid] for fid in faces])
        return min_layer
    
    # def _get_faces_obove_minimal_layer_that_the_line_crosses(self, line, min_layer, pos):
    #     faces_on_side = self._get_faces_on_side_of_line(line, pos)
    #     # print(f"Faces that exists on the {pos} side of the line: {faces_on_side}")
    #     if not faces_on_side:
    #         return set()
    #     # print(f" Minimal layer among these faces: {min_layer}")
    #     face_layer_map = self._face_layer_map()
    #     faces_above_min_layer = {fid for fid in faces_on_side if face_layer_map[fid] >= min_layer}
    #     # print("Faces above minimal layer that the line crosses:", faces_above_min_layer)
    #     return faces_above_min_layer

    # def _get_chained_faces(self, start_fids, line, pos):
    #     vids = set()
    #     last_vids = None
    #     fids = set(start_fids)
    #     vid_face_map = self._face_vertex_map()
    #     iteration = 0
    #     while vids != last_vids:
    #         # print(f"Interation {iteration}: fids={fids}, vids={vids}")
    #         last_vids = vids.copy()
    #         for fid in fids:
    #             position = self._get_vertices_of_face_position_to_line(fid, line)[pos]
    #             for vid in position:
    #                 if vid not in vids:
    #                     vids.add(vid)
    #                     # print(f" Added vertex {vid} from face {fid}")
    #         for vid in vids:
    #             faces_of_vid = vid_face_map[vid]
    #             for fid in faces_of_vid:
    #                 if fid not in fids:
    #                     fids.add(fid)
    #                     # print(f" Added face {fid} from vertex {vid}")
    #         min_layer = self._get_lowers_layer_in_faces(fids)
    #         faces_above_min_layer = self._get_faces_obove_minimal_layer_that_the_line_crosses(line, min_layer, pos)
    #         # print(f"Faces above minimal layer that the line crosses: {faces_above_min_layer}")
    #         fids = fids.union(faces_above_min_layer)
    #         iteration += 1
    #     return fids, vids
    
    def _sort_faces_by_layers(self, faces, reverse=True):
        fids = list(faces)
        face_layer_map = self._face_layer_map()
        faces_sorted_by_layers = sorted(fids, key=lambda fid: face_layer_map[fid], reverse=reverse)
        return faces_sorted_by_layers
    
    def _get_line_equation(self, v1_id, v2_id):
        p1, p2 = self.vertices[v1_id], self.vertices[v2_id]
        return get_line_equasion(p1, p2)

    def _reflect_vertex(self, v_id, line_eq):
        self._check_vid_exists(v_id)
        self.vertices[v_id] = reflect_point(self.vertices[v_id], line_eq)     

    def _get_edge_line_intersection(self, line, edge):
        v1_id, v2_id = edge
        p1, p2 = self.vertices[v1_id], self.vertices[v2_id]
        return segment_line_intersection(line, (p1, p2))

    def _cut_face_along_line(self, line, fid):        
        self._check_fid_exists(fid)

        face = self.faces[fid]
        vertices = set()
        for i in range(len(face)):
            v1_id = face[i]
            v2_id = face[(i + 1) % len(face)]
            edge =  tuple(sorted((v1_id, v2_id)))
            v1_id, v2_id = edge
            line_in_check = self._get_line_equation(v1_id, v2_id)
            if line == line_in_check:
                vertices.add(v1_id)
                vertices.add(v2_id)
            else:
                intersection_ratio = self._get_edge_line_intersection(line, edge)
                if intersection_ratio is None:
                    continue
                if intersection_ratio == 0:
                    vertices.add(v1_id)
                elif intersection_ratio == 1:
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
        
        return tuple(sorted(vertices))

    # def find_initial_faces(self, line, vertex_to_fold, pos_to_fold):
    #     faces = []
    #     for fid, face in self.faces.items():
    #         if vertex_to_fold in face:
    #             # print(f"face {face} includes vertex {vertex_to_fold}")
    #             position_map = self._vertices_positions_list(face, line)
    #             if len(position_map[pos_to_fold]) > 0 and len(position_map[-pos_to_fold]) > 0:
    #                 # print(f"face {face} is crossed by the line")
    #                 faces.append(fid)
    #     # print(f"Initial faces found: {faces}")
    #     return faces
    
    # def get_folding_list(self, edge, vertex_to_fold):

    def get_crease_faces(self, v1_id, v2_id):
        faces_with_both = []
        faces_with_one = []
        for fid, face in self.faces.items():
            if v1_id in face and v2_id in face:
                faces_with_both.append(fid)
            elif v1_id in face or v2_id in face:
                faces_with_one.append(fid)
        if len(faces_with_both) > 0:
            logging.debug(f"Found crease faces with both vertices: {[self.faces[fid] for fid in faces_with_both]}")
            return faces_with_both
        if len(faces_with_one) > 0:
            logging.debug(f"Found crease faces with one vertex: {[self.faces[fid] for fid in faces_with_one]}")
            return faces_with_one
        raise ValueError("No faces found containing either vertex of the edge.")
    
    # def split_faces_to_fold(self, line, edge, fids):

    #     v1_id, v2_id = edge
    #     face_layer_map = self._face_layer_map()
    #     fids_to_cut = sorted(fids, key=lambda fid: face_layer_map[fid], reverse=True)

    #     faces_map = {}
    #     for fid in fids_to_cut:

    #         if v1_id in self.faces[fid] and v2_id in self.faces[fid]:
    #             new_edge = edge
    #         else:
    #             new_edge = self._cut_face_along_line(line, fid)                    

    #         if new_edge:

    #             if self._edge_in_face(fid, new_edge):
    #                 raise ValueError("The crease line coincides with an edge of a face to fold.")

    #             new_v1_id, new_v2_id = new_edge

    #             face1, face2 = self._split_face(fid, new_v1_id, new_v2_id)
        
    #             new_fid = max(self.faces.keys()) + 1
    #             self.faces[fid] = face1
    #             self.faces[new_fid] = face2

    #             i = 0
    #             while i < len(face1) and (face1[i] == new_v1_id or face1[i] == new_v2_id):
    #                 i += 1
    #             face1_pos = self._get_vertex_position_to_line(face1[i], line)

    #             if face1_pos == 1:
    #                 positive_face = fid
    #                 negative_face = new_fid
    #             else:
    #                 positive_face = new_fid
    #                 negative_face = fid   

    #         else:
    #             face_pos = self._get_vertex_position_to_line(self.faces[fid][0], line)
    #             if face_pos == 1:
    #                 positive_face = fid
    #                 negative_face = None
    #             else:
    #                 positive_face = None
    #                 negative_face = fid 
                
    #             # self.faces_orientations[new_fid] = 1 - self.faces_orientations[fid] TODO

    #         faces_map[fid] = {
    #             'edge': new_edge,
    #             'layer': face_layer_map[fid],
    #             'positive_face': positive_face,
    #             'negative_face': negative_face,
    #         }
    #     return faces_map
    
    def _vids_position_map(self, line):
        vids_position_map = {}
        for vid in self.vertices.keys():
            position = self._get_vertex_position_to_line(vid, line)
            vids_position_map[vid] = position
        logging.debug(f"Vertices position map: {vids_position_map}")
        return vids_position_map

    def update_vertices_position_map(self, line, existing_map):
        new_vertices = set(self.vertices.keys()) - set(existing_map.keys())
        for vid in new_vertices:
            position = self._get_vertex_position_to_line(vid, line)
            existing_map[vid] = position
            logging.debug(f" Added new vertex to position map. Vertices position map: {existing_map}")
        return existing_map

    def fold_preparations(self, edge):
        logging.debug(f" Preparing to fold along edge {edge}")
        v1_id, v2_id = edge
        line = self._get_line_equation(v1_id, v2_id)
        vids_position_map = self._vids_position_map(line)
        fids_layer_map = self._face_layer_map()
        faces_split_info = {}
        for fid, face in self.faces.items():
            if len(self.vertices) > len(vids_position_map):
                vids_position_map = self.update_vertices_position_map(line, vids_position_map)
            logging.debug(f" Preparing face {fid}: {face}")
            faces_split_info[fid] = {"crease": None,
                                    "split": False,
                                    "lid": fids_layer_map[fid],
                                    "vertices": {1: set(), 0: set(), -1: set()},
                                    "faces": {1: None, -1: None}}
            for vid in face:
                position = vids_position_map[vid]
                faces_split_info[fid]["vertices"][position].add(vid)
            if v1_id in face and v2_id in face:
                faces_split_info[fid]['crease'] = tuple(sorted((v1_id, v2_id)))
            else:
                faces_split_info[fid]['crease'] = self._cut_face_along_line(line, fid)

            if len(faces_split_info[fid]["vertices"][1]) > 0 and len(faces_split_info[fid]["vertices"][-1]) > 0:
                faces_split_info[fid]['split'] = True
                crease_vid1, crease_vid2 = faces_split_info[fid]['crease']
                face1, face2 = self._split_face(fid, crease_vid1, crease_vid2)
                if list(faces_split_info[fid]["vertices"][1])[0] in face1:
                    faces_split_info[fid]["faces"][1] = face1
                    faces_split_info[fid]["faces"][-1] = face2
                else:
                    faces_split_info[fid]["faces"][1] = face2
                    faces_split_info[fid]["faces"][-1] = face1
                logging.debug(f"    Face {fid} will be split into two faces: {face1} and {face2}")

            elif len(faces_split_info[fid]["vertices"][1]) > 0:
                faces_split_info[fid]["faces"][1] = face
                logging.debug(f"    All vertices of face {fid} are on the positive side of the line.")

            elif len(faces_split_info[fid]["vertices"][-1]) > 0:
                faces_split_info[fid]["faces"][-1] = face
                logging.debug(f"    All vertices of face {fid} are on the negative side of the line.")
            else:
                raise ValueError("Face has no vertices on either side of the line.")
        return line, faces_split_info

    def _get_faces_to_fold(self, faces_split_info, pos_to_fold, min_layer):
        logging.info(f" Getting faces to fold on the {pos_to_fold} side above layer {min_layer}")
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
        pos_to_fold = self._get_vertex_position_to_line(vertex_to_fold, line)
        self.fold_on_crease_by_side(edge, pos_to_fold)

    def get_two_fold_options(self, edge):
        line, faces_split_info = self.fold_preparations(edge)
        v1_id, v2_id = edge
        initial_faces_to_fold = self.get_crease_faces(v1_id, v2_id)
        min_layer = self._get_min_layer_in_faces(initial_faces_to_fold)
        faces_to_fold_positive = self._get_faces_to_fold(faces_split_info, 1, min_layer)
        faces_to_fold_negative = self._get_faces_to_fold(faces_split_info, -1, min_layer)
        return line, faces_split_info, faces_to_fold_positive, faces_to_fold_negative

    def _updates_faces_after_fold(self, faces_split_info, faces_to_fold, pos_to_fold):
        logging.debug(f"Updating faces after splitting... (faces: {self.faces}) pos_to_fold: {pos_to_fold}")
        faces_to_fold = sorted(faces_to_fold, key=lambda fid: faces_split_info[fid]['lid'], reverse=True)
        logging.debug(f"Faces to fold sorted by layers: {faces_to_fold}")
        vids_to_reflect = set()
        logging.debug(f"Starting to fold faces...")
        for fid in faces_to_fold:
            logging.debug(f" Folding face {fid} in layer {faces_split_info[fid]['lid']} with split={faces_split_info[fid]['split']} and vertices positions: {faces_split_info[fid]['faces']}")
            info = faces_split_info[fid]
            if info['split']:
                new_fid = max(self.faces.keys()) + 1
                pos_to_stay = (-1) * pos_to_fold
                logging.debug(f" info['faces']: {info['faces']} pos_to_stay: {pos_to_stay}, pos_to_fold: {pos_to_fold}")
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
        return faces_split_info, vids_to_reflect

    def _update_layers_after_fold(self, faces_split_info, faces_to_fold):
        logging.debug(f"Updating layers after splitting... current layers: {[self.layers]} (faces: {self.faces})")
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
                new_layer = self._find_layer_for_new_face(old_layer, self.faces[new_fid])
            else:
                new_layer = self._find_layer_for_new_face(new_layer - 1, self.faces[new_fid])
            if info['split']:
                logging.debug(f"  Face {fid} {self.faces[fid]} is folded. New face ID {new_fid} will be in layer {new_layer}.")
            else:
                logging.debug(f"  Face {fid} {self.faces[fid]} is folded entirely. Old layer {old_layer}, Moving to layer {new_layer}. self.layers: {self.layers}")
                # self.layers[info['lid']].remove(fid)
            # if not info['split']:
                # lid = info['lid']
                # logging.debug(f"  Face {fid} {self.faces[fid]} is folded entirely. Removing from layer {lid}. self.layers: {self.layers}")
                # if lid in self.layers.keys() and fid in self.layers[lid]:
                    # self.layers[info['lid']].remove(fid)
            if new_layer in self.layers.keys():
                self.layers[new_layer].append(new_fid)
            else:
                self.layers[new_layer] = [new_fid]
            logging.debug(f" Moved face {new_fid} to layer {new_layer}. self.layers: {self.layers}")

    def _reflect_vertices_after_fold(self, vids_to_reflect, line):
        logging.debug(f"Vertices to reflect: {vids_to_reflect}")
        for vid in vids_to_reflect:
            self._reflect_vertex(vid, line)

    def execute_fold(self, line, faces_split_info, faces_to_fold, pos_to_fold):
        faces_split_info, vids_to_reflect = self._updates_faces_after_fold(faces_split_info, faces_to_fold, pos_to_fold)
        self._update_layers_after_fold(faces_split_info, faces_to_fold)
        self._reflect_vertices_after_fold(vids_to_reflect, line)
    
    def fold(self, edge, pos_to_fold):
        if pos_to_fold not in [1, -1]:
            raise ValueError("Position to fold must be 1 (left) or -1 (right).")
        self._note_fold(edge, pos_to_fold)
        line, faces_split_info, faces_to_fold_positive, faces_to_fold_negative = self.get_two_fold_options(edge)
        faces_to_fold = faces_to_fold_positive if pos_to_fold == 1 else faces_to_fold_negative
        self.execute_fold(line, faces_split_info, faces_to_fold, pos_to_fold)

    def fold_on_crease_by_side(self, edge, pos_to_fold):
        self._note_fold(edge, pos_to_fold)
        line, faces_split_info = self.fold_preparations(edge)
        v1_id, v2_id = edge
        logging.debug(f"############# Folding along edge {edge} to the {pos_to_fold} side of the line #############")
        if pos_to_fold == 0:
            raise ValueError("Vertex to fold lies on the crease line.")
        initial_faces_to_fold = self.get_crease_faces(v1_id, v2_id)
        logging.debug(f"Initial faces to fold: {initial_faces_to_fold}")
        min_layer = self._get_min_layer_in_faces(initial_faces_to_fold)
        logging.debug(f"Minimal layer among these faces: {min_layer}")
        faces_to_fold = self._get_faces_to_fold(faces_split_info, pos_to_fold, min_layer)
        logging.debug(f"Faces to fold before sorting, above layer {min_layer} and in {pos_to_fold} position: {faces_to_fold}")
        faces_to_fold = sorted(faces_to_fold, key=lambda fid: faces_split_info[fid]['lid'], reverse=True)
        logging.debug(f"Faces to fold sorted by layers: {faces_to_fold}")
        vids_to_reflect = set()
        logging.debug(f"Starting to fold faces...")
        for fid in faces_to_fold:
            logging.debug(f" Folding face {fid} in layer {faces_split_info[fid]['lid']} with split={faces_split_info[fid]['split']} and vertices positions: {faces_split_info[fid]['faces']}")
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
        logging.debug(f"Updating layers after splitting... current layers: {[self.layers]} (faces: {self.faces})")
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
                new_layer = self._find_layer_for_new_face(old_layer, self.faces[new_fid])
            else:
                new_layer = self._find_layer_for_new_face(new_layer - 1, self.faces[new_fid])
            if info['split']:
                logging.debug(f"  Face {fid} {self.faces[fid]} is folded. New face ID {new_fid} will be in layer {new_layer}.")
            else:
                logging.debug(f"  Face {fid} {self.faces[fid]} is folded entirely. Old layer {old_layer}, Moving to layer {new_layer}. self.layers: {self.layers}")
                # self.layers[info['lid']].remove(fid)
            # if not info['split']:
                # lid = info['lid']
                # logging.debug(f"  Face {fid} {self.faces[fid]} is folded entirely. Removing from layer {lid}. self.layers: {self.layers}")
                # if lid in self.layers.keys() and fid in self.layers[lid]:
                    # self.layers[info['lid']].remove(fid)
            if new_layer in self.layers.keys():
                self.layers[new_layer].append(new_fid)
            else:
                self.layers[new_layer] = [new_fid]
            logging.debug(f" Moved face {new_fid} to layer {new_layer}. self.layers: {self.layers}")
        logging.debug(f"Vertices to reflect: {vids_to_reflect}")
        for vid in vids_to_reflect:
            self._reflect_vertex(vid, line)


        
        


    # def fold_on_crease(self, edge, vertex_to_fold):

    #     self._note_fold(edge, vertex_to_fold)

    #     edge = tuple(sorted(edge))

    #     v1_id, v2_id = edge

    #     self._check_vid_exists(vertex_to_fold)
    #     self._check_vid_exists(v1_id)
    #     self._check_vid_exists(v2_id)
        
    #     line = self._get_line_equation(v1_id, v2_id)
        
    #     pos_to_fold = self._get_vertex_position_to_line(vertex_to_fold, line)

    #     if pos_to_fold == 'on':
    #         raise ValueError("Vertex to fold lies on the crease line.")

    #     faces_to_fold = self.get_crease_faces(v1_id, v2_id)
    #     print(faces_to_fold)
    #     # print("Looking for chained faces...")
    #     fids_to_cut, vids_to_reflect = self._get_chained_faces(faces_to_fold, line, pos_to_fold)

    #     fids = list(fids_to_cut)

    #     faces_map =self.split_faces_to_fold(line, edge, fids)
        
    #     for vid in vids_to_reflect:
    #         self._reflect_vertex(vid, line)        
        
    #     for fid in fids_to_cut:

    #         info = faces_map[fid]
    #         old_layer = info['layer']
    #         folded_face = info['folded_face']
    #         if self.faces[folded_face] == [7, 4, 9]:
    #             new_layer = self._find_layer_for_new_face(old_layer, self.faces[folded_face], debug=True)
    #         else:
    #             new_layer = self._find_layer_for_new_face(old_layer, self.faces[folded_face])
            
    #         if info['edge'] is None:
    #             self.layers[old_layer].remove(fid)

    #         if new_layer in self.layers.keys():
    #             self.layers[new_layer].append(folded_face)
    #         else:
    #             self.layers[new_layer] = [folded_face]

    
    def split_face_by_edge_and_ratio(self, edge, ratio_list=0.5):
        
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
    
    def _get_all_edges(self):
        edges = set()
        for fid in self.faces.keys():
            face_edges = self._edges_of_face(fid)
            edges.update(face_edges)
        return edges
    
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
        
    def _remove_vid(self, v_id):
        max_vid = max(self.vertices.keys())
        self.vertices[v_id] = self.vertices[max_vid]
        del self.vertices[max_vid]
        for fid, face in self.faces.items():
            if max_vid in face:
                face[face.index(max_vid)] = v_id
        return max_vid

    def pocket(self, pivot_vid, shifted_vid, upper_vid, lower_vid):
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
        
        # if not np.allclose(self.vertices[upper_vid], self.vertices[lower_vid]):
        #     raise ValueError("Upper and lower vertices must be at the same position.")
        
        # # Split the lower face
        # lower_face, new_lower_vid = self.split_pocket_face(pivot_vid, shared_vid, lower_vid)
        # lower_face1, lower_face2 = self.split_face(lower_face, shared_vid, new_lower_vid)
        # if lower_vid in lower_face1:
        #     staying_lower_face = lower_face1
        # else:
        #     staying_lower_face = lower_face2
        # self.faces[lower_face] = staying_lower_face

        # # Split the upper face
        # upper_face, new_upper_vid = self.split_pocket_face(pivot_vid, shared_vid, upper_vid)





    # def pocket_faces(self, pivot_point, origin_edge_second_point, target_edge_second_point):
    #     upper_face = self._find_face_with_3_points(pivot_point, origin_edge_second_point, target_edge_second_point)
    #     if not self._edge_in_face(upper_face, tuple(sorted((pivot_point, origin_edge_second_point)))):
    #         raise ValueError("The pivot point and origin edge second point do not form an edge in the upper face.")
    #     if upper_face is None:
    #         raise ValueError("No face contains the three specified points.")
        
    #     lower_face = self._find_face_with_2_points(origin_edge_second_point, target_edge_second_point)
    #     if lower_face is None:
    #         raise ValueError("No face contains the two specified points.")
    #     if not self._edge_in_face(lower_face, tuple(sorted((pivot_point, origin_edge_second_point)))):
    #         raise ValueError("The pivot point and origin edge second point do not form an edge in the lower face.")

    #     faces_layers = self._face_layer_map()
    #     if faces_layers[upper_face] - faces_layers[lower_face] != 1:    
    #         raise ValueError("The upper face must be above the lower face.")
    
    #     return upper_face, lower_face
    
    # def pocket(self, pivot_point, origin_edge_second_point, target_edge_second_point):
    #     upper_face, lower_face = self.pocket_faces(pivot_point, origin_edge_second_point, target_edge_second_point)
    #     upper1, upper2 = self.split_face(upper_face, pivot_point, target_edge_second_point)
    #     line = self._get_line_equation(pivot_point, target_edge_second_point)
    #     edge = self._cut_face_along_line(line, lower_face)
    #     lower1, lower2 = self.split_face(lower_face, edge[0], edge[1])
    #     new_fid = max(self.faces.keys()) + 1
        

    def plot_layers(self):
        cols = 3
        rows = math.ceil(len(self.layers) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
        axes = axes.flatten()
        for i, layer in enumerate(reversed(self.layers.values())):
            for fid in layer:
                face = self.faces[fid]
                coords = [self.vertices[vid] for vid in face]
                x_coords = [p[0] for p in coords]
                y_coords = [p[1] for p in coords]
                orientation = self.faces_orientations.get(fid, 0)
                color = 'white' if orientation == 0 else 'lightblue'
                axes[i].fill(x_coords, y_coords, color=color, alpha=0.5, edgecolor='none', zorder=5)
                len_face = len(face)
                for j in range(len_face):
                    p1, p2 = self.vertices[face[j]], self.vertices[face[(j + 1) % len_face]]
                    axes[i].plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=1.5, zorder=5)
            axes[i].set_xlim(-2, 12)
            axes[i].set_ylim(-2, 12)
        plt.show()


    def plot(self, show_vertices_indices=True, show_all_edges=True, print_layers=False):
        """Plots the current state of the origami paper with filled faces."""
        fig, ax = plt.subplots(figsize=(4, 4))
        
        if show_all_edges:
            all_edges = self._get_all_edges()
            for edge in all_edges:
                p1, p2 = self.vertices[edge[0]], self.vertices[edge[1]]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=0.3, zorder=7)

        def vertices_by_pos():
            pos_dict = {}
            for vid, pos in self.vertices.items():
                found_point_in_dict = False
                for point in pos_dict.keys():
                    if np.allclose(point, pos, atol=1e-5):
                        pos_dict[point].append(vid)
                        found_point_in_dict = True
                        break
                if not found_point_in_dict:
                    pos_dict[tuple(pos)] = [vid]
            return pos_dict
        
        if show_vertices_indices:
            pos_dict = vertices_by_pos()
            for pos, vids in pos_dict.items():
                for i, vid in enumerate(vids):
                    ax.text(pos[0], pos[1] - (i * 0.5) - 0.2, str(vid), ha='center', fontsize=9, va='bottom',
                            bbox=dict(boxstyle="circle,pad=0.05", fc="white", alpha=1, ec="black", lw=0.5), zorder=10)
        
        faces_to_plot = self._sort_faces_by_layers(self.faces.keys(), reverse=False)
        for face_id in faces_to_plot:
            face = self.faces[face_id]

            len_face = len(face)

            coords = [self.vertices[vid] for vid in face]
            orientation = self.faces_orientations.get(face_id, 0)
            color = 'white' if orientation == 0 else 'lightblue'
            x_coords = [p[0] for p in coords]
            y_coords = [p[1] for p in coords]

            ax.fill(x_coords, y_coords, color=color, alpha=1, edgecolor='none', zorder=5)

            for i in range(len_face):
                p1, p2 = self.vertices[face[i]], self.vertices[face[(i + 1) % len_face]]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=1.5, zorder=5)
            

        ax.set_xlim(-2, 12)
        ax.set_ylim(-2, 12)

        if print_layers:
            print(".... Faces by Layers ...")
            for layer in sorted(self.layers.keys()):
                print(f"    Layer {layer}")
                for fid in self.layers[layer]:
                    print(f"        Face {fid}: {self.faces[fid]} Orientation: {self.faces_orientations[fid]}")

        plt.show()
