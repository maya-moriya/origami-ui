import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import math
from src.utils import *

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

    def _note_fold(self, edge, vertex):
        self.actions.append({"action": "fold", "edge": edge, "vertex": vertex})
    
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
        lid = old_lid + 1
        while True:
            if lid not in self.layers.keys():
                return lid
            if self._face_overlap_with_layer(self.layers[lid], new_face):
                lid += 1
            else:
                return lid
    
    def _add_face_to_layer(self, fid, lid):
        self._check_fid_exists(fid)
        if lid in self.layers.keys():
            self.layers[lid].append(fid)
        else:
            self.layers[lid] = [fid]
    
    def _vertices_position_map_to_line(self, vertices, line):
        position_map = {1: set(), -1: set(), 0: set()}
        for vid in vertices:
            position = self._get_vertex_position_to_line(vid, line)
            position_map[position].add(vid)
        return position_map
    
    def _get_vertices_of_face_position_to_line(self, fid, line):
        self._check_fid_exists(fid)
        face = self.faces[fid]
        return self._vertices_position_map_to_line(face, line)

    def _get_faces_on_side_of_line(self, line, side):
        if side not in [1, -1]:
            raise ValueError("Side must be 1 (left) or -1 (right).")
        faces_on_side = set()
        for fid in self.faces.keys():
            position_map = self._get_vertices_of_face_position_to_line(fid, line)
            if len(position_map[side]) > 0:
                faces_on_side.add(fid)
        return faces_on_side
    
    def _get_lowers_layer_in_faces(self, faces):
        face_layer_map = self._face_layer_map()
        min_layer = min([face_layer_map[fid] for fid in faces])
        return min_layer
    
    def _get_faces_obove_minimal_layer_that_the_line_crosses(self, line, min_layer, pos):
        faces_on_side = self._get_faces_on_side_of_line(line, pos)
        print(f"Faces that exists on the {pos} side of the line: {faces_on_side}")
        if not faces_on_side:
            return set()
        print(f" Minimal layer among these faces: {min_layer}")
        face_layer_map = self._face_layer_map()
        faces_above_min_layer = {fid for fid in faces_on_side if face_layer_map[fid] >= min_layer}
        print("Faces above minimal layer that the line crosses:", faces_above_min_layer)
        return faces_above_min_layer

    def _get_chained_faces(self, start_fids, line, pos):
        vids = set()
        last_vids = None
        fids = set(start_fids)
        vid_face_map = self._face_vertex_map()
        iteration = 0
        while vids != last_vids:
            print(f"Interation {iteration}: fids={fids}, vids={vids}")
            last_vids = vids.copy()
            for fid in fids:
                position = self._get_vertices_of_face_position_to_line(fid, line)[pos]
                for vid in position:
                    if vid not in vids:
                        vids.add(vid)
                        print(f" Added vertex {vid} from face {fid}")
            for vid in vids:
                faces_of_vid = vid_face_map[vid]
                for fid in faces_of_vid:
                    if fid not in fids:
                        fids.add(fid)
                        print(f" Added face {fid} from vertex {vid}")
            min_layer = self._get_lowers_layer_in_faces(fids)
            faces_above_min_layer = self._get_faces_obove_minimal_layer_that_the_line_crosses(line, min_layer, pos)
            print(f"Faces above minimal layer that the line crosses: {faces_above_min_layer}")
            fids = fids.union(faces_above_min_layer)
            iteration += 1
        return fids, vids
    
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
    

    def find_initial_faces(self, line, vertex_to_fold, pos_to_fold):
        faces = []
        for fid, face in self.faces.items():
            if vertex_to_fold in face:
                print(f"face {face} includes vertex {vertex_to_fold}")
                position_map = self._vertices_position_map_to_line(face, line)
                if len(position_map[pos_to_fold]) > 0 and len(position_map[-pos_to_fold]) > 0:
                    print(f"face {face} is crossed by the line")
                    faces.append(fid)
        print(f"Initial faces found: {faces}")
        return faces

    def fold_on_crease(self, edge, vertex_to_fold):

        self._note_fold(edge, vertex_to_fold)

        edge = tuple(sorted(edge))

        v1_id, v2_id = edge

        self._check_vid_exists(vertex_to_fold)
        self._check_vid_exists(v1_id)
        self._check_vid_exists(v2_id)
        
        line = self._get_line_equation(v1_id, v2_id)
        
        fold_within_face = False
        face_to_fold = self._find_face_crossed_by_edge_and_containing_vertex(edge, vertex_to_fold)
        if face_to_fold is not None:
            print(f"Folding within face {face_to_fold}")
            fold_within_face = True

            if self._edge_in_face(face_to_fold, edge):
                raise ValueError("The crease line coincides with an edge of the face to fold.")
        else:
            face_to_fold = self._find_face_containing_one_side_of_the_edge_and_vertex(edge, vertex_to_fold)
            if face_to_fold is not None:
                print(f"Folding starting from face {face_to_fold} (includes one side of the edge and the vertex to fold)")

        
        pos_to_fold = self._get_vertex_position_to_line(vertex_to_fold, line)

        if pos_to_fold == 'on':
            raise ValueError("Vertex to fold lies on the crease line.")


        if face_to_fold is not None:
            faces_to_fold = [face_to_fold]
            print(f"Face to fold: {face_to_fold}, vertices: {self.faces[face_to_fold]}")
        else:
            print(f" No face contains both the edge and the vertex to fold. Finding initial faces...")
            faces_to_fold = self.find_initial_faces(line, vertex_to_fold, 1)

        print("Looking for chained faces...")
        fids_to_cut, vids_to_reflect = self._get_chained_faces(faces_to_fold, line, pos_to_fold)

        fids = list(fids_to_cut)
        face_layer_map = self._face_layer_map()
        fids_to_cut = sorted(fids, key=lambda fid: face_layer_map[fid], reverse=True)

        faces_map = {}
        for fid in fids_to_cut:
            if fold_within_face and (fid == face_to_fold):
                new_edge = edge
            else:
                new_edge = self._cut_face_along_line(line, fid)                    

            if new_edge is not None:

                if self._edge_in_face(fid, new_edge):
                    raise ValueError("The crease line coincides with an edge of a face to fold.")

                new_v1_id, new_v2_id = new_edge

                face1, face2 = self._split_face(fid, new_v1_id, new_v2_id)
        
                i = 0
                while i < len(face1) and (face1[i] == new_v1_id or face1[i] == new_v2_id):
                    i += 1
                face_pos = self._get_vertex_position_to_line(face1[i], line)

                if face_pos == pos_to_fold:
                    folded_face = face1
                    staying_face = face2
                else:
                    folded_face = face2
                    staying_face = face1   
                
                new_fid = max(self.faces.keys()) + 1

                self.faces[fid] = staying_face
                self.faces[new_fid] = folded_face
                
                self.faces_orientations[new_fid] = 1 - self.faces_orientations[fid]

                faces_map[fid] = {
                    'edge': new_edge,
                    'layer': face_layer_map[fid],
                    'staying_face': fid,
                    'folded_face': new_fid,
                }
            
            else:
                faces_map[fid] = {
                    'edge': None,
                    'staying_face': None,
                    'layer': face_layer_map[fid],
                    'folded_face': fid
                }
                self.faces_orientations[fid] = 1 - self.faces_orientations[fid]

        for vid in vids_to_reflect:
            self._reflect_vertex(vid, line)        
        
        for fid in fids_to_cut:

            info = faces_map[fid]
            old_layer = info['layer']
            folded_face = info['folded_face']
            if self.faces[folded_face] == [7, 4, 9]:
                new_layer = self._find_layer_for_new_face(old_layer, self.faces[folded_face], debug=True)
            else:
                new_layer = self._find_layer_for_new_face(old_layer, self.faces[folded_face])
            
            if info['edge'] is None:
                self.layers[old_layer].remove(fid)

            if new_layer in self.layers.keys():
                self.layers[new_layer].append(folded_face)
            else:
                self.layers[new_layer] = [folded_face]

    
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
