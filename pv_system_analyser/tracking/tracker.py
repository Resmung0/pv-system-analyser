from itertools import count

import numpy as np
from src.scripts.tracking.frame import FrameHandler
from src.scripts.tracking.matcher import PointMatcher

# TODO: Usar `skimage.measure.manders_overlap_coeff` para calcular a overlap entre imagens.


class Tracker:
    def __init__(
        self,
        n_frames: int,
        n_keypoints: int = 500,
        overlap_threshold: float = 0.5,
    ) -> None:
        """
        Parameters
        ----------
        n_frames : int
            Number of frames in the video.
        n_keypoints : int, optional
            Number of ORB keypoints to extract from frames. By default 500.
        overlap_threshold : float, optional
            Threshold value (range from 0 to 1) used to quantify overlap of sequencial frames.
            If the percent of tracked rois from frame t + 1 is bellow this value, new id's
            will be given to them. By default 0.5.
        """
        self.frame_handler = FrameHandler(n_keypoints)
        self.point_matcher = PointMatcher()

        self.rois = {f"frame_{frame + 1}": None for frame in range(n_frames)}
        self.overlap_threshold = overlap_threshold
        self.frames_overlap = []
        self.__id_generator = count(1)

    def update(
        self, frame: np.ndarray, detected_centroids: list[int]
    ) -> dict[str, list[int]]:
        rois_tracked = dict()
        prev_frame_id = self.frame_handler.previous_frame_info["id"]

        # Handle rois from the first frame
        if prev_frame_id == 0:
            rois = self.__compact_new_rois(detected_centroids)
            rois_tracked.update(rois)

        # Handle rois from the others frames
        else:
            projected_centroids = self.frame_handler.project_points(
                frame, detected_centroids
            )
            self.point_matcher.match(detected_centroids, projected_centroids)

            # Handles objects of non-overlaping sequencial frames (assign new ids)
            if self._check_frame_overlap_rate(detected_centroids):
                rois = self.__compact_new_rois(detected_centroids)
                rois_tracked.update(rois)

            # Handles objects with overlaping sequencial frames
            else:
                # Handle objects who still on the scene
                rois_tracked = self._handle_old_objs(
                    detected_centroids, rois_tracked, prev_frame_id
                )
                # Handle objects who enter the scene
                rois = self.__compact_new_rois(
                    detected_centroids, self.point_mathcher.unmatches
                )
                rois_tracked.update(rois)

        self.rois[f"frame_{prev_frame_id}"] = rois_tracked

    def _handle_old_objs(self, detected_centroids, rois_tracked, prev_frame_id):
        prev_tracked_rois = self.rois[f"frame_{prev_frame_id}"]
        for detected_id, tracked_id in self.point_matcher.matches:
            for key, value in prev_tracked_rois.items():
                prev_frame_centroids = self.frame_handler.previous_frame_info[
                    "centroids"
                ][tracked_id]
                if np.all(prev_frame_centroids == value):
                    rois_tracked[key] = detected_centroids[detected_id]
        return rois_tracked

    def _check_frame_overlap_rate(self, detected_centroids):
        """Method that checks if the overlap rate between the
        current frame and the previous one is less than the threshold.
        """
        percent_tracked_objs = (
            self.point_matcher.matches.shape[0] / detected_centroids.shape[0]
        )
        self.frames_overlap.append(percent_tracked_objs)
        return percent_tracked_objs <= self.overlap_threshold

    def __compact_new_rois(self, rois, rois_ids=None):
        """
        Function that generates new ID's to tracked ROI's.
        """
        if rois_ids is not None:
            rois = rois[rois_ids]
        return {next(self.__id_generator): roi for roi in rois}
