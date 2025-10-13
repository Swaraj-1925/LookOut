import os
import cv2
import json
import numpy as np
from pathlib import Path
from datetime import timedelta
from insightface.app import FaceAnalysis

class FaceRecognitionPipeline:
    """
    Enhanced pipeline for CCTV footage that combines:
    1. Face recognition (when clear faces available)
    2. Whole-person appearance matching (hair, glasses, clothing, body shape)
    3. Multi-feature verification
    """

    def __init__(self, ctx_id=0, det_size=(640, 640)):
        print("Initializing Enhanced CCTV Recognition Pipeline...")

        # Face recognition (primary method for clear footage)
        self.app = FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

        print("✓ Pipeline initialized")

    def extract_query_features(self, query_image_path):
        """
        Extract MULTIPLE features from query image:
        1. Face embedding (if available)
        2. Full appearance features (hair, clothing, body)
        3. Color histogram
        4. Visual descriptors
        """
        img = cv2.imread(query_image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {query_image_path}")

        features = {
            'face_embedding': None,
            'face_bbox': None,
            'appearance_features': None,
            'color_histogram': None,
            'has_glasses': False,
            'dominant_colors': None,
            'image_shape': img.shape
        }

        # 1. Try to extract face embedding
        faces = self.app.get(img)
        if len(faces) > 0:
            if len(faces) > 1:
                print(f"Warning: {len(faces)} faces found, using the largest one")
                faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)

            query_face = faces[0]
            features['face_embedding'] = query_face.normed_embedding
            features['face_bbox'] = query_face.bbox.astype(int).tolist()

            # Detect glasses (check if landmarks near eyes have occlusion)
            features['has_glasses'] = self._detect_glasses(query_face)

            print(f"✓ Face embedding extracted: {features['face_embedding'].shape}")
        else:
            print("⚠️ No face detected in query - will use appearance-only matching")

        # 2. Extract appearance features (ALWAYS, even if face found)
        features['appearance_features'] = self._extract_appearance_features(img)

        # 3. Color histogram (for clothing/appearance matching)
        features['color_histogram'] = self._extract_color_histogram(img)

        # 4. Extract dominant colors
        features['dominant_colors'] = self._extract_dominant_colors(img)

        print(f"✓ Multi-modal features extracted")
        print(f"  - Face: {'Yes' if features['face_embedding'] is not None else 'No'}")
        print(f"  - Glasses detected: {features['has_glasses']}")
        print(f"  - Dominant colors: {len(features['dominant_colors'])} colors")

        return features

    def _detect_glasses(self, face):
        """Simple glasses detection based on landmarks"""
        # This is a placeholder
        return False  # Set to True if you know person has glasses

    def _extract_appearance_features(self, img):
        """
        Extract appearance features using SIFT/ORB for texture matching
        These capture hair style, clothing patterns, etc.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Use ORB (faster than SIFT, patent-free)
        orb = cv2.ORB_create(nfeatures=500)
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        if descriptors is not None:
            # Create compact representation
            # Average pooling of descriptors
            avg_descriptor = np.mean(descriptors, axis=0)
            return avg_descriptor

        return None

    def _extract_color_histogram(self, img, bins=32):
        """
        Extract color histogram for clothing/appearance matching
        """
        # Convert to HSV (more robust to lighting changes)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Calculate histogram for each channel
        h_hist = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [bins], [0, 256])

        # Normalize
        h_hist = cv2.normalize(h_hist, h_hist).flatten()
        s_hist = cv2.normalize(s_hist, s_hist).flatten()
        v_hist = cv2.normalize(v_hist, v_hist).flatten()

        # Concatenate
        color_hist = np.concatenate([h_hist, s_hist, v_hist])

        return color_hist

    def _extract_dominant_colors(self, img, k=5):
        """Extract dominant colors for clothing matching"""
        # Reshape image to list of pixels
        pixels = img.reshape(-1, 3).astype(np.float32)

        # Apply K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Convert to list
        dominant_colors = centers.astype(int).tolist()

        return dominant_colors

    def compare_color_histograms(self, hist1, hist2):
        """Compare two color histograms using correlation"""
        if hist1 is None or hist2 is None:
            return 0.0

        # Use correlation method (returns value between -1 and 1)
        correlation = cv2.compareHist(
            hist1.astype(np.float32),
            hist2.astype(np.float32),
            cv2.HISTCMP_CORREL
        )

        # Normalize to 0-1
        return (correlation + 1) / 2

    def compare_appearance_features(self, feat1, feat2):
        """Compare appearance features using cosine similarity"""
        if feat1 is None or feat2 is None:
            return 0.0

        # Cosine similarity
        similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))

        return max(0, similarity)  # Clip to [0, 1]

    def multi_modal_match(self, query_features, frame_features, weights=None):
        """
        Multi-modal matching that combines:
        - Face similarity (if available)
        - Appearance similarity
        - Color histogram similarity

        Args:
            query_features: Features from query image
            frame_features: Features from video frame
            weights: Dict with keys 'face', 'appearance', 'color'

        Returns:
            is_match: Boolean
            scores: Dict with individual scores
            final_score: Weighted average
        """
        if weights is None:
            # Default weights (adjust based on your needs)
            weights = {
                'face': 0.5,        # Face gets 50% if available
                'appearance': 0.3,   # Appearance gets 30%
                'color': 0.2        # Color gets 20%
            }

        scores = {}
        valid_weights = {}

        # 1. Face similarity (only if both have faces)
        if (query_features.get('face_embedding') is not None and
            frame_features.get('face_embedding') is not None):
            face_sim = np.dot(
                query_features['face_embedding'],
                frame_features['face_embedding']
            )
            scores['face'] = float(face_sim)
            valid_weights['face'] = weights['face']
        else:
            scores['face'] = None

        # 2. Appearance similarity
        app_sim = self.compare_appearance_features(
            query_features.get('appearance_features'),
            frame_features.get('appearance_features')
        )
        scores['appearance'] = float(app_sim)
        valid_weights['appearance'] = weights['appearance']

        # 3. Color histogram similarity
        color_sim = self.compare_color_histograms(
            query_features.get('color_histogram'),
            frame_features.get('color_histogram')
        )
        scores['color'] = float(color_sim)
        valid_weights['color'] = weights['color']

        # Calculate weighted average (only for valid scores)
        total_weight = sum(valid_weights.values())
        if total_weight == 0:
            final_score = 0.0
        else:
            # Normalize weights
            normalized_weights = {k: v/total_weight for k, v in valid_weights.items()}

            # Calculate weighted sum
            final_score = sum(
                scores[k] * normalized_weights[k]
                for k in normalized_weights.keys()
                if scores[k] is not None
            )

        # Decision logic - more lenient for CCTV
        # If face is available, require reasonable face match + good appearance/color
        # If no face, rely on appearance + color
        if scores['face'] is not None:
            # Face + appearance-based decision
            is_match = (
                (scores['face'] >= 0.35 and final_score >= 0.40) or  # Lowered thresholds
                (scores['face'] >= 0.30 and scores['appearance'] >= 0.6 and scores['color'] >= 0.6)
            )
        else:
            # Appearance-only decision (more lenient)
            is_match = (
                (scores['appearance'] >= 0.55 and scores['color'] >= 0.50) or
                final_score >= 0.55
            )

        return is_match, scores, final_score

    def detect_and_match_persons(self, frame, query_features, frame_count,
                                  timestamp, match_count, output_folder):
        """
        Detect all persons in frame and match against query
        """
        try:
            # Extract features from entire frame (for appearance matching)
            frame_features = {
                'face_embedding': None,
                'appearance_features': self._extract_appearance_features(frame),
                'color_histogram': self._extract_color_histogram(frame)
            }

            # Try face detection
            faces = self.app.get(frame)

            # If faces found, check each one
            for face_idx, face in enumerate(faces):
                frame_features['face_embedding'] = face.normed_embedding

                # Multi-modal matching
                is_match, scores, final_score = self.multi_modal_match(
                    query_features,
                    frame_features
                )

                if is_match:
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox

                    # Extract face crop
                    face_crop = frame[y1:y2, x1:x2]

                    # Save matched face
                    timestamp_str = str(timedelta(seconds=timestamp)).replace(':', '-')
                    filename = f"match_{match_count:04d}_frame{frame_count:06d}_{timestamp_str}_score{final_score:.3f}.jpg"
                    filepath = os.path.join(output_folder, filename)
                    cv2.imwrite(filepath, face_crop)

                    # Store match info with ALL scores
                    match_info = {
                        'frame_number': frame_count,
                        'timestamp': timestamp,
                        'timestamp_formatted': str(timedelta(seconds=timestamp)),
                        'bbox': {'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)},
                        'final_score': float(final_score),
                        'face_similarity': scores['face'],
                        'appearance_similarity': scores['appearance'],
                        'color_similarity': scores['color'],
                        'confidence': float(face.det_score),
                        'match_type': 'multi_modal',
                        'filename': filename
                    }

                    return match_info

            # If no faces detected, try whole-frame matching (for distant/blurry shots)
            if len(faces) == 0:
                is_match, scores, final_score = self.multi_modal_match(
                    query_features,
                    frame_features
                )

                # More strict threshold for no-face matching
                if is_match and final_score >= 0.65:
                    h, w = frame.shape[:2]
                    timestamp_str = str(timedelta(seconds=timestamp)).replace(':', '-')
                    filename = f"match_{match_count:04d}_frame{frame_count:06d}_{timestamp_str}_noface_score{final_score:.3f}.jpg"
                    filepath = os.path.join(output_folder, filename)
                    cv2.imwrite(filepath, frame)

                    match_info = {
                        'frame_number': frame_count,
                        'timestamp': timestamp,
                        'timestamp_formatted': str(timedelta(seconds=timestamp)),
                        'bbox': {'x1': 0, 'y1': 0, 'x2': w, 'y2': h},
                        'final_score': float(final_score),
                        'face_similarity': None,
                        'appearance_similarity': scores['appearance'],
                        'color_similarity': scores['color'],
                        'confidence': 0.0,
                        'match_type': 'appearance_only',
                        'filename': filename
                    }

                    return match_info

            return None

        except Exception as e:
            print(f"Error processing frame {frame_count}: {str(e)}")
            return None

    def process_video_and_find_person(self, video_path, query_features, output_folder,
                                      frame_skip=1, save_all_detections=False):
        """
        Process video with enhanced multi-modal matching
        """
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        print(f"\n{'='*60}")
        print(f"ENHANCED CCTV PROCESSING")
        print(f"{'='*60}")
        print(f"Total frames: {total_frames}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps:.2f}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Multi-modal matching: Face + Appearance + Color")
        print(f"{'='*60}\n")

        matches = []
        frame_count = 0
        processed_count = 0
        match_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            processed_count += 1
            timestamp = frame_count / fps

            # Detect and match
            match_info = self.detect_and_match_persons(
                frame, query_features, frame_count, timestamp,
                match_count, output_folder
            )

            if match_info:
                matches.append(match_info)
                match_count += 1
                print(f"✓ Match found at {match_info['timestamp_formatted']} "
                      f"(Score: {match_info['final_score']:.3f}, Type: {match_info['match_type']})")

            if processed_count % 50 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% | Matches: {match_count}")

            frame_count += 1

        cap.release()

        # Save metadata
        metadata = {
            'video_info': {
                'fps': float(fps),
                'total_frames': total_frames,
                'width': width,
                'height': height,
                'duration': float(duration)
            },
            'query_features': {
                'has_face': query_features['face_embedding'] is not None,
                'has_glasses': query_features['has_glasses'],
                'num_dominant_colors': len(query_features['dominant_colors'])
            },
            'matches': matches,
            'statistics': {
                'total_frames_processed': processed_count,
                'total_matches_found': match_count,
                'match_rate': match_count / processed_count if processed_count > 0 else 0
            }
        }

        metadata_path = os.path.join(output_folder, 'matches_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total matches: {match_count}")
        print(f"Match rate: {match_count/processed_count*100:.2f}%")
        print(f"Metadata saved: {metadata_path}")
        print(f"{'='*60}\n")

        return matches
