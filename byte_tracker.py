import numpy as np

class ByteTracker:
    """
    ByteTrack algorithm for robust multi-object tracking
    Tracks both high and low confidence detections
    """

    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8):
        """
        Initialize ByteTrack

        Args:
            track_thresh: High confidence threshold
            track_buffer: Number of frames to keep lost tracks
            match_thresh: IoU threshold for matching
        """
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.frame_id = 0
        self.track_id_count = 0

    def update(self, detections, features=None):
        """
        Update tracks with new detections

        Args:
            detections: List of detection dicts with bbox, confidence, features
            features: Optional list of feature vectors

        Returns:
            active_tracks: List of active track dicts
        """
        self.frame_id += 1

        # Separate high and low confidence detections
        high_dets = [d for d in detections if d['confidence'] >= self.track_thresh]
        low_dets = [d for d in detections if d['confidence'] < self.track_thresh]

        # Match high confidence detections with existing tracks
        matched, unmatched_tracks, unmatched_dets = self._match_detections(
            self.tracked_tracks, high_dets, features
        )

        # Update matched tracks
        for track_idx, det_idx in matched:
            track = self.tracked_tracks[track_idx]
            det = high_dets[det_idx]
            track['bbox'] = det['bbox']
            track['confidence'] = det['confidence']
            track['features'] = det.get('features')
            track['age'] = 0
            track['hits'] += 1

        # Initialize new tracks from unmatched high confidence detections
        for det_idx in unmatched_dets:
            det = high_dets[det_idx]
            new_track = {
                'track_id': self.track_id_count,
                'bbox': det['bbox'],
                'confidence': det['confidence'],
                'features': det.get('features'),
                'age': 0,
                'hits': 1,
                'reid_similarity': det.get('reid_similarity', 0.0)
            }
            self.tracked_tracks.append(new_track)
            self.track_id_count += 1

        # Move unmatched tracks to lost
        for track_idx in unmatched_tracks:
            track = self.tracked_tracks[track_idx]
            track['age'] += 1
            if track['age'] <= self.track_buffer:
                self.lost_tracks.append(track)

        # Remove matched tracks
        self.tracked_tracks = [t for i, t in enumerate(self.tracked_tracks)
                               if i not in unmatched_tracks]

        # Second matching with low confidence detections
        if low_dets and self.lost_tracks:
            matched_lost, _, _ = self._match_detections(
                self.lost_tracks, low_dets, None, match_thresh=0.5
            )

            for track_idx, det_idx in matched_lost:
                track = self.lost_tracks[track_idx]
                det = low_dets[det_idx]
                track['bbox'] = det['bbox']
                track['age'] = 0
                self.tracked_tracks.append(track)

            # Clean up lost tracks
            self.lost_tracks = [t for i, t in enumerate(self.lost_tracks)
                               if i not in [m[0] for m in matched_lost]]

        # Remove old lost tracks
        self.lost_tracks = [t for t in self.lost_tracks if t['age'] <= self.track_buffer]

        return self.tracked_tracks

    def _match_detections(self, tracks, detections, features=None, match_thresh=None):
        """
        Match tracks to detections using IoU and appearance similarity
        """
        if match_thresh is None:
            match_thresh = self.match_thresh

        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        # Compute cost matrix (IoU + appearance similarity)
        cost_matrix = np.zeros((len(tracks), len(detections)))

        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                # IoU similarity
                iou = self._compute_iou(track['bbox'], det['bbox'])

                # Appearance similarity (if features available)
                appearance_sim = 0.0
                if track.get('features') is not None and det.get('features') is not None:
                    appearance_sim = np.dot(track['features'], det['features'])

                # Combined cost (higher is better)
                cost_matrix[i, j] = 0.7 * iou + 0.3 * appearance_sim

        # Hungarian matching
        matched_indices = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(detections)))

        # Simple greedy matching
        for _ in range(min(len(tracks), len(detections))):
            if cost_matrix.size == 0:
                break

            max_val = cost_matrix.max()
            if max_val < match_thresh:
                break

            i, j = np.unravel_index(cost_matrix.argmax(), cost_matrix.shape)
            matched_indices.append((i, j))

            if i in unmatched_tracks:
                unmatched_tracks.remove(i)
            if j in unmatched_dets:
                unmatched_dets.remove(j)

            cost_matrix[i, :] = -1
            cost_matrix[:, j] = -1

        return matched_indices, unmatched_tracks, unmatched_dets

    def _compute_iou(self, box1, box2):
        """Compute IoU between two boxes"""
        x1_inter = max(box1['x1'], box2['x1'])
        y1_inter = max(box1['y1'], box2['y1'])
        x2_inter = min(box1['x2'], box2['x2'])
        y2_inter = min(box1['y2'], box2['y2'])

        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        box1_area = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
        box2_area = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])

        iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
        return iou

