import math
import numpy as np

def get_line_equasion(p1, p2):
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    A = y2 - y1
    B = x1 - x2
    C = -A * x1 - B * y1
    norm = math.hypot(A, B)
    if norm > 0:
        A /= norm
        B /= norm
        C /= norm
        if abs(A) > 1e-12:
            if A < 0:
                A, B, C = -A, -B, -C
        else:
            if B < 0:
                A, B, C = -A, -B, -C
    else:
        A, B, C = 0.0, 0.0, 0.0
    return A, B, C

def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0
    
def point_side_to_line(point, line):
    A, B, C = line
    d = A * point[0] + B * point[1] + C
    EPSILON = 1e-5
    if abs(d) < EPSILON:
        return 0
    return sign(d)

def segment_line_intersection(line, segment):
    p1, p2 = segment
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    A, B, C = line
    EPSILON = 1e-5
    D = A * dx + B * dy
    if math.fabs(D) < EPSILON:
        return None
    N = - (A * x1 + B * y1 + C)
    t = N / D
    if t > 0 and t < EPSILON:
        return 0
    if t < 1 and t > 1 - EPSILON:
        return 1
    return t

def reflect_point(point, line_eq):
    p_x, p_y = point
    A, B, C = line_eq
    denominator = A**2 + B**2
    if denominator == 0: return np.array([p_x, p_y])
    t = -2 * (A * p_x + B * p_y + C) / denominator
    return np.array([p_x + A * t, p_y + B * t])

def get_overlap(min1, max1, min2, max2, eps=1e-12):
    """Checks for 1D overlap between two ranges [min1, max1] and [min2, max2]."""
    return max(min1, min2) <= min(max1, max2)
    # return max(min1, min2) < min(max1, max2) - eps

def project_on_axis(points, axis):
    """
    Projects all points of a shape onto a given axis and returns the
    minimum and maximum values of the projections.
    """
    dot_products = []

    for x, y in points:
        dot_products.append(x * axis[0] + y * axis[1])
        
    return min(dot_products), max(dot_products)


def do_faces_overlap(face1, face2):
    """
    Checks if two **convex** polygons (faces) overlap using the 
    Separating Axis Theorem (SAT).
    
    :param face1: List of points [(x1, y1), ..., (xN, yN)] for the first polygon.
    :param face2: List of points [(x1, y1), ..., (xM, yM)] for the second polygon.
    :return: True if the faces overlap, False otherwise.
    """
        
    all_faces = [face1, face2]
    
    for points in all_faces:

        num_vertices = len(points)

        for i in range(num_vertices):
            p1 = points[i]
            p2 = points[(i + 1) % num_vertices]

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]

            axis = (-dy, dx)

            min1, max1 = project_on_axis(face1, axis)
            min2, max2 = project_on_axis(face2, axis)

            if not get_overlap(min1, max1, min2, max2):
                return False

    return True

def are_collinear(p1, p2, p3, tolerance=1e-9):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    area_determinant = (x1 * (y2 - y3) + 
                        x2 * (y3 - y1) + 
                        x3 * (y1 - y2))
    return abs(area_determinant) < tolerance

def find_angle_bisector_ratio(p1, p2, p3):
    P1 = np.array(p1)
    P2 = np.array(p2)
    P3 = np.array(p3)
    side_a_length = np.linalg.norm(P3 - P2)
    side_c_length = np.linalg.norm(P1 - P2)
    if side_c_length == 0 and side_a_length == 0:
        raise ValueError("Points p1, p2, and p3 are identical; angle bisector is undefined.")
    elif side_c_length == 0:
        raise ValueError("Points p1 and p2 are identical; angle bisector is undefined.")
    elif side_a_length == 0:
        raise ValueError("Points p2 and p3 are identical; angle bisector is undefined.")
    total_length = side_a_length + side_c_length
    t = side_c_length / total_length
    # P4 = (1 - t) * P1 + t * P3
    return 1 - t