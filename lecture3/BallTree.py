import numpy as np
import pandas as pd

def calc_minkowski_distance(vect, p, axis=None):
    return np.pow(np.sum(np.abs(vect) ** p, axis=axis), 1/p)

class BallNode:
    def __init__(
        self,
        points,
        indexes,
        p,
        metric,
        leaf_size
    ):
        self.points = points
        self.indexes = indexes
        self.leaf_size = leaf_size
        self.left = None
        self.right = None

        self.centroid = np.mean(points, axis=0)

        distances_to_centroid = calc_minkowski_distance(points - self.centroid, p, axis=1)
        furthest_point_index_from_center = np.argmax(distances_to_centroid)
        point_A = points[furthest_point_index_from_center]

        # radius is distance from the center to the furthest point
        self.radius = distances_to_centroid[furthest_point_index_from_center]

        if len(points) <= leaf_size:
            return
            
        distances_to_point_A = calc_minkowski_distance(points - point_A, p, axis=1)
        furthest_point_index_from_point_A = np.argmax(distances_to_point_A)
        point_B = points[furthest_point_index_from_point_A] 

        distances_to_point_B = calc_minkowski_distance(points - point_B, p, axis=1)

        # get points indexes which are closer to point A than point B 
        left_mask = distances_to_point_A <= distances_to_point_B
        
        # get points indexes which are closer to point B than point A
        right_mask = ~left_mask

        """
        if all points are the same, point A and point B will be the same.
        'left_mask = distances_to_point_A <= distances_to_point_B' results in including
        all points in left_group. The right group will be empty. On the next recursive calls
        the situation will be the same. It will repeat infinitely. It will be sad
        """
        if np.any(left_mask) and np.any(right_mask):
            self.left = BallNode(
                points=points[left_mask], 
                indexes=indexes[left_mask], 
                leaf_size=leaf_size,
                metric=metric,
                p=p
            )
            self.right = BallNode(
                points=points[right_mask],
                indexes=indexes[right_mask], 
                leaf_size=leaf_size,
                metric=metric,
                p=p
            )
            self.points = None

    def is_leaf(self):
        return self.points is not None

class MyBallTree:
    def __init__(
            self, 
            points, 
            leaf_size=1,
            metric="minkowski",
            p=2
    ):
        indexes = np.arange(len(points))
        self.metric = metric
        self.p = p
        self.root = BallNode(
            points=np.asarray(points), 
            indexes=indexes, 
            leaf_size=leaf_size,
            metric=metric,
            p=p
        )

    def query(self, X, k=5):
        X_arr = np.asarray(X, dtype=np.float64)
        
        all_neighbours = []
        
        for target_p in X_arr:
            target_p = np.asarray(target_p)
            
            # Array<Tuple<distance_between_target_and_neighbour, neighbour_point_coordinates>>
            best_neighbours = [(float("inf"), None) for _ in range(k)]
    
            def search(node):
                if node is None:
                    return
                    
                distance_between_target_and_center = calc_minkowski_distance(target_p - node.centroid, self.p)
    
                # in other words, distance between the target point and the closest point on the surface of the sphere 
                min_distance_between_target_and_any_point_in_ball = distance_between_target_and_center - node.radius
    
                furthest_between_best_neighbours_distance = best_neighbours[-1][0]
    
                # ignore the sphere if the furthest neighbour (among already checked) is
                # closer to the target point than the closest point in the sphere 
                if furthest_between_best_neighbours_distance <= min_distance_between_target_and_any_point_in_ball:
                    return
    
                if node.is_leaf():
                    for curr_point_index, curr_point_dist in enumerate(calc_minkowski_distance(node.points - target_p, self.p, axis=1)):
                        furthest_between_best_neighbours_distance = best_neighbours[-1][0]
                        
                        if curr_point_dist < furthest_between_best_neighbours_distance:
                            global_point_index = node.indexes[curr_point_index]
                            best_neighbours[-1] = (curr_point_dist, global_point_index)
                            best_neighbours.sort(key=lambda x: x[0])
                    return
                        
                dist_between_left_centroid_and_target = calc_minkowski_distance(node.left.centroid - target_p, self.p)
                dist_between_right_centroid_and_target = calc_minkowski_distance(node.right.centroid - target_p, self.p)
    
                if dist_between_left_centroid_and_target <= dist_between_right_centroid_and_target:
                    search(node.left)
                    search(node.right)
                else:
                    search(node.right)
                    search(node.left)
            search(self.root)
            all_neighbours.append(best_neighbours)

        res = ([], [])
        
        for idx, best_neighbours in enumerate(all_neighbours):
            res[0].append([])
            res[1].append([])
            for dist, index in best_neighbours:
                res[0][idx].append(dist)
                res[1][idx].append(index)
            
        return res
