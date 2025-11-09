import cv2
import numpy as np
import json
import os
from byte_tracker import ByteTracker
from collections import defaultdict, deque
from datetime import timedelta

class VideoTracker:
    """
    Enhanced video tracker with ByteTrack and temporal filtering
    """

    def __init__(self, reid_pipeline, body_pipeline, face_pipeline):
        """
        Initialize enhanced tracker
        """
        self.reid = reid_pipeline
        self.body_detector = body_pipeline
        self.face_detector = face_pipeline

        # Initialize ByteTrack
        self.bytetrack = ByteTracker(
            track_thresh=0.6,
            track_buffer=30,
            match_thresh=0.7
        )
        # Temporal consistency buffer
        self.detection_history = deque(maxlen=10)  # Last 10 frames

        print("Video Tracker initialized with ByteTrack")

    def verify_with_face(self, frame, bbox, query_embedding, threshold=0.5):
        """
        Additional verification using face recognition

        Args:
            frame: Video frame
            bbox: Person bounding box
            query_embedding: Query person's face embedding
            threshold: Face similarity threshold

        Returns:
            is_match: True if face matches
            face_similarity: Similarity score
        """
        try:
            # Extract person region
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            person_crop = frame[y1:y2, x1:x2]

            if person_crop.size == 0:
                return False, 0.0

            # Detect face in person region
            faces = self.face_detector.app.get(person_crop)

            if len(faces) == 0:
                return False, 0.0

            # Get largest face
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            face_embedding = face.normed_embedding

            # Compare with query
            similarity = np.dot(query_embedding, face_embedding)

            return similarity >= threshold, float(similarity)

        except Exception as e:
            return False, 0.0

    def temporal_consistency_filter(self, detections, min_appearances=3):
        """
        Filter detections based on temporal consistency

        Args:
            detections: Current frame detections
            min_appearances: Minimum appearances in history to confirm

        Returns:
            filtered_detections: Consistent detections only
        """
        self.detection_history.append(detections)

        if len(self.detection_history) < min_appearances:
            return detections

        # Count appearances across history
        consistent_dets = []
        for det in detections:
            appearances = 0
            for hist_frame in self.detection_history:
                for hist_det in hist_frame:
                    iou = self._compute_iou(det['bbox'], hist_det['bbox'])
                    if iou > 0.3:  # Same person region
                        appearances += 1
                        break

            if appearances >= min_appearances:
                consistent_dets.append(det)

        return consistent_dets

    def match_person_in_frame(self, frame, gallery, query_embedding=None,
                              reid_threshold=0.6, face_verification=True):
        """
        Enhanced person matching with multi-level verification
        """
        # Detect all persons
        results = self.body_detector.person_detector(frame, conf=0.3, classes=[0], verbose=False)

        if len(results) == 0 or len(results[0].boxes) == 0:
            return []

        detections = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            body_crop = frame[y1:y2, x1:x2]

            if body_crop.size == 0:
                continue

            # Extract appearance features
            features = self.reid.extract_reid_features(body_crop)

            if features is None:
                continue

            # Compare with gallery
            similarities = []
            for gallery_item in gallery[:10]:  # Top 10 samples
                sim = self.reid.compute_feature_similarity(features, gallery_item['features'])
                similarities.append(sim)

            max_similarity = max(similarities) if similarities else 0.0
            avg_similarity = np.mean(similarities) if similarities else 0.0

            # First level: Appearance matching
            if max_similarity >= reid_threshold:

                # Second level: Face verification (if enabled)
                face_match = face_verification
                face_sim = 0.0

                if face_verification and query_embedding is not None:
                    face_match, face_sim = self.verify_with_face(
                        frame, {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                        query_embedding, threshold=0.5
                    )

                # Combine scores
                final_score = max_similarity
                if face_match:
                    final_score = 0.6 * max_similarity + 0.4 * face_sim  # Boost if face matches

                detections.append({
                    'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                    'confidence': float(box.conf[0]),
                    'reid_similarity': float(max_similarity),
                    'face_similarity': float(face_sim),
                    'face_verified': face_match,
                    'final_score': float(final_score),
                    'features': features['osnet_features'] if 'osnet_features' in features else None
                })

        return detections

    def track_person_in_video(self, video_path, gallery, output_path, query_image_path=None,
                              query_embedding=None, reid_threshold=0.6, frame_skip=1,
                              use_face_verification=True, use_temporal_filter=True):
        """
        tracking with ByteTrack and multi-level verification
        """
        print(f"\n{'='*60}")
        print(f"TRACKING WITH BYTETRACK")
        print(f"{'='*60}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video: {width}x{height} @ {fps:.2f} FPS")
        print(f"Total frames: {total_frames}")
        print(f"Re-ID threshold: {reid_threshold}")
        print(f"Face verification: {'ON' if use_face_verification else 'OFF'}")
        print(f"Temporal filtering: {'ON' if use_temporal_filter else 'OFF'}")
        print(f"{'='*60}\n")

        # Load query thumbnail
        query_thumbnail = None
        if query_image_path and os.path.exists(query_image_path):
            query_img = cv2.imread(query_image_path)
            if query_img is not None:
                thumb_size = 150
                h, w = query_img.shape[:2]
                if h > w:
                    new_h = thumb_size
                    new_w = int(w * (thumb_size / h))
                else:
                    new_w = thumb_size
                    new_h = int(h * (thumb_size / w))
                query_thumbnail = cv2.resize(query_img, (new_w, new_h))
                border_size = 3
                query_thumbnail = cv2.copyMakeBorder(
                    query_thumbnail, border_size, border_size, border_size, border_size,
                    cv2.BORDER_CONSTANT, value=[0, 255, 0]
                )

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        tracking_results = []
        frame_count = 0
        detection_count = 0
        target_track_id = None  # Track ID of our target person

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_count / fps

            if frame_count % frame_skip == 0:
                # Detect and match persons
                detections = self.match_person_in_frame(
                    frame, gallery, query_embedding,
                    reid_threshold=reid_threshold,
                    face_verification=use_face_verification
                )

                # Apply temporal filtering
                if use_temporal_filter and len(detections) > 0:
                    detections = self.temporal_consistency_filter(detections)

                # Update ByteTrack
                tracks = self.bytetrack.update(detections)

                # Find target track (highest Re-ID score)
                target_track = None
                if tracks:
                    # Sort by final_score or reid_similarity
                    valid_tracks = [t for t in tracks if t.get('reid_similarity', 0) >= reid_threshold]
                    if valid_tracks:
                        target_track = max(valid_tracks, key=lambda t: t.get('reid_similarity', 0))
                        if target_track_id is None:
                            target_track_id = target_track['track_id']
                        elif target_track['track_id'] == target_track_id:
                            # Continue tracking same ID
                            pass
                        else:
                            # Check if should switch to higher confidence track
                            current_track = next((t for t in tracks if t['track_id'] == target_track_id), None)
                            if current_track is None or target_track['reid_similarity'] > current_track.get('reid_similarity', 0) + 0.1:
                                target_track_id = target_track['track_id']

                # Draw target track
                if target_track:
                    detection_count += 1
                    bbox = target_track['bbox']
                    reid_sim = target_track.get('reid_similarity', 0.0)
                    face_verified = target_track.get('face_verified', False)

                    # Draw bounding box
                    color = (0, 255, 0) if face_verified else (0, 200, 200)
                    cv2.rectangle(frame, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), color, 4)

                    # Corner markers
                    corner_len = 20
                    thickness = 6
                    for corner in [(bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y1']),
                                   (bbox['x1'], bbox['y2']), (bbox['x2'], bbox['y2'])]:
                        cv2.circle(frame, corner, 8, color, -1)

                    # Label
                    label = f"TARGET #{target_track['track_id']}"
                    conf_label = f"Match: {reid_sim:.1%}"
                    verify_label = "✓ Face Verified" if face_verified else "Body Match Only"

                    label_y = bbox['y1'] - 60
                    cv2.rectangle(frame, (bbox['x1'], label_y), (bbox['x1'] + 200, bbox['y1']), (0, 150, 0), -1)
                    cv2.putText(frame, label, (bbox['x1'] + 5, label_y + 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, conf_label, (bbox['x1'] + 5, label_y + 40),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, verify_label, (bbox['x1'] + 5, bbox['y1'] - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0) if face_verified else (255, 255, 0), 2)

                    tracking_results.append({
                        'frame_number': frame_count,
                        'timestamp': timestamp,
                        'track_id': target_track['track_id'],
                        'bbox': bbox,
                        'reid_similarity': reid_sim,
                        'face_verified': face_verified
                    })

            # Add query thumbnail
            if query_thumbnail is not None:
                margin = 20
                thumb_h, thumb_w = query_thumbnail.shape[:2]
                y1, y2 = margin, margin + thumb_h
                x1, x2 = width - thumb_w - margin, width - margin

                overlay = frame.copy()
                cv2.rectangle(overlay, (x1-10, y1-10), (x2+10, y2+40), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                frame[y1:y2, x1:x2] = query_thumbnail
                cv2.putText(frame, "QUERY PERSON", (x1-10, y2+25),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Info overlay
            info_text = [
                f"Frame: {frame_count}/{total_frames}",
                f"Time: {str(timedelta(seconds=int(timestamp)))}",
                f"Detections: {detection_count}",
                f"Track ID: {target_track_id if target_track_id else 'None'}"
            ]

            y_offset = 30
            for i, text in enumerate(info_text):
                cv2.putText(frame, text, (20, y_offset + i*30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, text, (20, y_offset + i*30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)

            out.write(frame)

            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% | Frame: {frame_count}/{total_frames} | Detections: {detection_count}")

            frame_count += 1

        cap.release()
        out.release()

        # Save metadata
        metadata_path = output_path.replace('.mp4', '_tracking.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                'total_frames': frame_count,
                'frames_with_detections': detection_count,
                'detection_rate': detection_count / frame_count if frame_count > 0 else 0,
                'target_track_id': target_track_id,
                'tracking_data': tracking_results
            }, f, indent=2)

        print(f"\n{'='*60}")
        print(f"TRACKING COMPLETE")
        print(f"{'='*60}")
        print(f"Frames processed: {frame_count}")
        print(f"Frames with detection: {detection_count}")
        print(f"Detection rate: {detection_count/frame_count*100:.1f}%")
        print(f"Target Track ID: {target_track_id}")
        print(f"Output video: {output_path}")
        print(f"Metadata: {metadata_path}")
        print(f"{'='*60}\n")

        return tracking_results

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

