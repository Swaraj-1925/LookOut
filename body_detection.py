import cv2
import json
import os
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
import matplotlib.pyplot as plt

class BodyDetectionPipeline:
    """
    Phase 2: Extract full body from face detections for missing person identification
    Methods: ROI-based person detection, pose estimation, tracking
    """

    def __init__(self, person_model='yolov8x.pt', pose_model='yolov8x-pose.pt'):
        """
        Initialize body detection pipeline

        Args:
            person_model: YOLO model for person detection (yolov8n/s/m/l/x)
            pose_model: YOLO model for pose estimation
        """
        print("Initializing Body Detection Pipeline...")

        # Load person detector
        self.person_detector = YOLO(person_model)
        print(f"✓ Person detector loaded: {person_model}")

        # Load pose estimator
        self.pose_estimator = YOLO(pose_model)
        print(f"✓ Pose estimator loaded: {pose_model}")

        self.tracker_initialized = False
        self.tracker = None

    def expand_face_to_roi(self, face_bbox, frame_shape, k_w=3, k_h=8):
        x1, y1, x2, y2 = face_bbox['x1'], face_bbox['y1'], face_bbox['x2'], face_bbox['y2']

        face_w = x2 - x1
        face_h = y2 - y1
        face_cx = (x1 + x2) // 2
        face_cy = (y1 + y2) // 2

        # Expand region
        roi_w = int(face_w * k_w)
        roi_h = int(face_h * k_h)

        # Center around face but shift up slightly (face is in upper body)
        roi_x1 = max(0, face_cx - roi_w // 2)
        roi_y1 = max(0, face_cy - int(roi_h * 0.15))  # Face is ~15% from top
        roi_x2 = min(frame_shape[1], roi_x1 + roi_w)
        roi_y2 = min(frame_shape[0], roi_y1 + roi_h)

        return {'x1': roi_x1, 'y1': roi_y1, 'x2': roi_x2, 'y2': roi_y2}

    def detect_person_in_roi(self, frame, roi_bbox, face_bbox, conf_threshold=0.3):
        """
        Method 1: Run person detector only in ROI (efficient)

        Args:
            frame: Full video frame
            roi_bbox: Region of interest around face
            face_bbox: Original face bounding box
            conf_threshold: Detection confidence threshold

        Returns:
            body_bbox: Best matching person bounding box or None
        """
        # Extract ROI
        roi = frame[roi_bbox['y1']:roi_bbox['y2'], roi_bbox['x1']:roi_bbox['x2']]

        # Run person detector on ROI
        results = self.person_detector(roi, conf=conf_threshold, classes=[0], verbose=False)

        if len(results) == 0 or len(results[0].boxes) == 0:
            return None

        # Find person box that contains face center
        face_cx = (face_bbox['x1'] + face_bbox['x2']) // 2
        face_cy = (face_bbox['y1'] + face_bbox['y2']) // 2

        best_person = None
        best_iou = 0

        for box in results[0].boxes:
            # Convert ROI coordinates to frame coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            person_x1 = int(x1) + roi_bbox['x1']
            person_y1 = int(y1) + roi_bbox['y1']
            person_x2 = int(x2) + roi_bbox['x1']
            person_y2 = int(y2) + roi_bbox['y1']

            # Check if face center is inside person box
            if person_x1 <= face_cx <= person_x2 and person_y1 <= face_cy <= person_y2:
                # Calculate IoU between face and person box
                iou = self._calculate_iou(face_bbox,
                    {'x1': person_x1, 'y1': person_y1, 'x2': person_x2, 'y2': person_y2})

                if iou > best_iou:
                    best_iou = iou
                    best_person = {
                        'x1': person_x1, 'y1': person_y1,
                        'x2': person_x2, 'y2': person_y2,
                        'confidence': float(box.conf[0]),
                        'method': 'roi_detector'
                    }

        return best_person

    def estimate_body_from_pose(self, frame, roi_bbox, face_bbox, conf_threshold=0.3):
        """
        Method 2: Use pose estimation to get precise body boundaries

        Args:
            frame: Full video frame
            roi_bbox: Region of interest
            face_bbox: Original face bounding box
            conf_threshold: Keypoint confidence threshold

        Returns:
            body_bbox: Body bounding box from keypoints or None
        """
        # Extract ROI
        roi = frame[roi_bbox['y1']:roi_bbox['y2'], roi_bbox['x1']:roi_bbox['x2']]

        # Run pose estimation
        results = self.pose_estimator(roi, conf=conf_threshold, verbose=False)

        if len(results) == 0 or results[0].keypoints is None:
            return None

        # Find pose that corresponds to our face
        face_cx = (face_bbox['x1'] + face_bbox['x2']) // 2
        face_cy = (face_bbox['y1'] + face_bbox['y2']) // 2

        best_pose = None
        min_dist = float('inf')

        for idx, keypoints in enumerate(results[0].keypoints.xy):
            kpts = keypoints.cpu().numpy()

            # Check if we have valid keypoints
            if len(kpts) < 17:  # COCO format has 17 keypoints
                continue

            # Nose keypoint (index 0) should be near face
            nose = kpts[0]
            if nose[0] == 0 and nose[1] == 0:  # Invalid keypoint
                continue

            nose_x = int(nose[0]) + roi_bbox['x1']
            nose_y = int(nose[1]) + roi_bbox['y1']

            dist = np.sqrt((nose_x - face_cx)**2 + (nose_y - face_cy)**2)
            if dist < min_dist:
                min_dist = dist
                best_pose = kpts

        if best_pose is None:
            return None

        # Extract body box from keypoints
        # COCO keypoints: 0-nose, 5-left_shoulder, 6-right_shoulder,
        # 11-left_hip, 12-right_hip, 13-left_knee, 14-right_knee
        valid_points = []
        for i in [0, 5, 6, 11, 12, 13, 14]:  # Key body points
            if i < len(best_pose) and not (best_pose[i][0] == 0 and best_pose[i][1] == 0):
                valid_points.append(best_pose[i])

        if len(valid_points) < 3:  # Need at least 3 points
            return None

        valid_points = np.array(valid_points)

        # Calculate bounding box with padding
        pad = 20
        x1 = int(np.min(valid_points[:, 0])) + roi_bbox['x1'] - pad
        y1 = int(np.min(valid_points[:, 1])) + roi_bbox['y1'] - pad
        x2 = int(np.max(valid_points[:, 0])) + roi_bbox['x1'] + pad
        y2 = int(np.max(valid_points[:, 1])) + roi_bbox['y1'] + pad

        # Clip to frame boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        return {
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'keypoints': valid_points,
            'method': 'pose_estimation'
        }

    def heuristic_body_estimation(self, face_bbox, frame_shape):
        """
        Fallback Method: Simple heuristic expansion (when detectors fail)

        Args:
            face_bbox: Face bounding box
            frame_shape: Frame dimensions

        Returns:
            body_bbox: Estimated body box
        """
        x1, y1, x2, y2 = face_bbox['x1'], face_bbox['y1'], face_bbox['x2'], face_bbox['y2']

        face_w = x2 - x1
        face_h = y2 - y1
        face_cx = (x1 + x2) // 2

        # Typical body proportions (face is ~1/8 of total height)
        body_w = int(face_w * 2.5)
        body_h = int(face_h * 8)

        # Face is typically at ~12% from top of body
        body_x1 = max(0, face_cx - body_w // 2)
        body_y1 = max(0, y1 - int(body_h * 0.12))
        body_x2 = min(frame_shape[1], body_x1 + body_w)
        body_y2 = min(frame_shape[0], body_y1 + body_h)

        return {
            'x1': body_x1, 'y1': body_y1,
            'x2': body_x2, 'y2': body_y2,
            'method': 'heuristic'
        }

    def get_body_from_face(self, frame, face_bbox):
        """
        Main method: Extract full body from face detection

        Args:
            frame: Video frame
            face_bbox: Face bounding box dict

        Returns:
            body_bbox: Body bounding box dict with method used
        """
        frame_shape = frame.shape

        # Try ROI detector first (best tradeoff)
        roi_bbox = self.expand_face_to_roi(face_bbox, frame_shape, k_w=3, k_h=8)
        body = self.detect_person_in_roi(frame, roi_bbox, face_bbox)

        if body is not None:
            return body

        # Try pose estimation
        body = self.estimate_body_from_pose(frame, roi_bbox, face_bbox)

        if body is not None:
            return body

        # Fallback to heuristic
        return self.heuristic_body_estimation(face_bbox, frame_shape)

    def process_matched_faces(self, video_path, matches_metadata_path, output_folder, strategy='hybrid'):
        """
        Process all matched faces from Phase 1 to extract full bodies

        Args:
            video_path: Path to video file
            matches_metadata_path: JSON file from Phase 1
            output_folder: Output folder for body crops
            strategy: Body detection strategy ('hybrid', 'roi_detector', 'pose', 'heuristic')

        Returns:
            body_results: List of body detections with metadata
        """
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        # Load Phase 1 results
        with open(matches_metadata_path, 'r') as f:
            metadata = json.load(f)

        # Handle both old and new metadata formats
        if 'matches' in metadata:
            matches = metadata['matches']
            video_info = metadata.get('video_info', {})
        else:
            # Old format - direct list of matches
            matches = metadata
            video_info = {}

        if not matches:
            print("No matches found in metadata")
            return []

        print(f"\n{'='*60}")
        print(f"PHASE 2: BODY DETECTION")
        print(f"{'='*60}")
        print(f"Total face matches to process: {len(matches)}")
        print(f"Detection strategy: {strategy}")
        if video_info:
            print(f"Video FPS: {video_info.get('fps', 'N/A')}")
            print(f"Video duration: {video_info.get('duration', 'N/A'):.2f}s")
        print(f"{'='*60}\n")

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        body_results = []
        method_counts = defaultdict(int)

        for idx, match in enumerate(matches):
            frame_num = match['frame_number']
            face_bbox = match['bbox']

            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()

            if not ret:
                print(f"Warning: Could not read frame {frame_num}")
                continue

            # Get body bounding box
            body_bbox = self.get_body_from_face(frame, face_bbox)

            if body_bbox:
                # Extract body crop
                body_crop = frame[body_bbox['y1']:body_bbox['y2'],
                                 body_bbox['x1']:body_bbox['x2']]

                # Save body image
                timestamp_str = match['timestamp_formatted'].replace(':', '-')
                filename = f"body_{idx:04d}_frame{frame_num:06d}_{timestamp_str}_{body_bbox['method']}.jpg"
                filepath = os.path.join(output_folder, filename)
                cv2.imwrite(filepath, body_crop)

                # Store results
                body_result = {
                    'frame_number': frame_num,
                    'timestamp': match['timestamp'],
                    'timestamp_formatted': match['timestamp_formatted'],
                    'face_bbox': face_bbox,
                    'body_bbox': {k: int(v) if isinstance(v, (int, np.integer)) else v
                                  for k, v in body_bbox.items() if k != 'keypoints'},
                    'detection_method': body_bbox['method'],
                    'filename': filename,
                    'similarity': match['similarity']
                }

                body_results.append(body_result)
                method_counts[body_bbox['method']] += 1

            if (idx + 1) % 20 == 0:
                print(f"Processed: {idx + 1}/{len(matches)}")

        cap.release()

        # Save metadata
        metadata_path = os.path.join(output_folder, 'body_detections_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(body_results, f, indent=2)

        print(f"\n{'='*60}")
        print(f"BODY DETECTION COMPLETE")
        print(f"{'='*60}")
        print(f"Total bodies extracted: {len(body_results)}")
        print(f"Detection methods used:")
        for method, count in method_counts.items():
            print(f"  - {method}: {count}")
        print(f"Output folder: {output_folder}")
        print(f"Metadata saved: {metadata_path}")
        print(f"{'='*60}\n")

        return body_results

    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        x1_inter = max(box1['x1'], box2['x1'])
        y1_inter = max(box1['y1'], box2['y1'])
        x2_inter = min(box1['x2'], box2['x2'])
        y2_inter = min(box1['y2'], box2['y2'])

        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        box1_area = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
        box2_area = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])

        iou = inter_area / (box1_area + box2_area - inter_area)
        return iou
