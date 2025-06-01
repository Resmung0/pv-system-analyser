from skimage.feature import ORB, match_descriptors
from skimage.transform import ProjectiveTransform
from skimage.measure import ransac


class FrameHandler:
    """Frame handler class."""
    
    def __init__(self, n_keypoints: int) -> None:
        self.feature_extractor = ORB(n_keypoints=n_keypoints)
        self.motion_detector = ProjectiveTransform()
        self.previous_frame_info = {
            "keypoints": None,
            "descriptors": None,
            "centroids": None,
            "id": 0
        }
    
    
    def project_points(self, current_frame, roi_points):
        """
        Project ROI's centroids from previous frame to current frame.
        First, a feature extractor extract keypoints and descriptors 
        from current frame. Then, these information are matched with
        the ones from previous frame to update the motion detector. 
        Finally, the motion detector is used to project the ROI's 
        centroids on current frame.
        """        
        # Extract keypoints and descriptors from current frame
        self.feature_extractor.detect_and_extract(current_frame)
        
        # Match them with previous frame information
        frames_matched_points = self._match_frames()
        
        # Save current frame descriptors and keypoints as old ones
        self.__update_previous_frame_info(roi_points)
        self.frame_id += 1
        
        # Estimate ROI points position in the current frame
        self.__update_motion_detector(frames_matched_points)
        return self.motion_detector(roi_points)

    
    def _match_frames(self):
        """
        Match current and previous frames descriptors by brute force to 
        obtain matched keypoints between them.
        """
        matches = match_descriptors(
            self.previous_frame_info['descriptors'], self.feature_extractor.descriptors,
        )        
        matched_keypoints = (
            self.previous_frame_info['keypoints'][matches[:, 0]],
            self.feature_extractor.keypoints[matches[:, 1]]
        )
        return matched_keypoints
    
    def __update_motion_detector(self, points: tuple[int, int]) -> None:
        self.motion_detector, _ = ransac(
            points,
            self.motion_detector,
            min_samples=8,
            residual_threshold=2,
            max_trials=5000,
        )
    
    def __update_previous_frame_info(self, points) -> None:
        self.previous_frame_info['keypoints'] = self.feature_extractor.keypoints
        self.previous_frame_info['descriptors'] = self.feature_extractor.descriptors
        self.previous_frame_info['centroids'] = points
    