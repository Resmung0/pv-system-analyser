from scipy.spatial import KDTree
import numpy as np


class PointMatcher:
    def __init__(self, max_distance: int = 500) -> None:
        self.ref_distance = max_distance
        self.unmatches, self.matches = [], []
    
    def match(
        self, 
        detected_points: list[int],
        tracked_points: list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        distances, detected_ids = self.__calculate_points_distances(
            detected_points, tracked_points
        )
        self._check_unmatches(detected_points, detected_ids)
        detected_ids = self._remove_duplicated_ids(detected_ids, distances)
        self._check_matches(detected_ids, distances)
        
        self.unmatches = np.array(sorted(self.unmatches, reverse=True))
        self.matches = np.array(self.matches)
        
    
    def _check_matches(self, detected_ids, distances):
        ids_distances = zip(detected_ids, distances)
        for track_id, (detected_id, distance) in enumerate(ids_distances):
            if detected_id != -1:
                if distance <= self.ref_distance:
                    self.matches.append([detected_id, track_id])
                else:
                    self.unmatches.append(detected_id)
    
    def _check_unmatches(self, detected_points, detected_ids):
        for detected_id, _ in enumerate(detected_points):
            if detected_id not in detected_ids:
                self.unmatches.append(detected_id)
            
    @staticmethod
    def _remove_duplicates_from_ids(ids, distances):
        labels, counts = np.unique(ids, return_counts=True)
        duplicates = labels[counts > 1]
        for duplicate in duplicates:
            duplicated_ids = np.argwhere(ids == duplicate)
            duplicated_distances = distances[duplicated_ids]
            keep_id = np.argmin(duplicated_distances)
            remove_ids = np.delete(duplicated_ids, keep_id)
            ids[remove_ids] = -1
        return ids
    
    @staticmethod
    def __calculate_points_distances(points_1, points_2):
        return KDTree(points_1).query(points_2)
    