import os
import cv2
import json
import numpy as np
from pathlib import Path
from datetime import timedelta
from insightface.app import FaceAnalysis

class FaceRecognitionPipeline:

    def __init__(self, ctx_id=0, det_size=(640, 640)):
        print("Initializing Face Recognition Pipeline...")

        # Initialize InsightFace app (includes RetinaFace detector + ArcFace recognizer)
        self.app = FaceAnalysis(
                name='buffalo_l',  # Uses ResNet-100 backbone
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

        print("✓ Pipeline initialized")

    def extract_query_embedding(self, query_image_path):
        """
        Step 1-3 for query image: Detect → Align → Extract embedding

        Returns:
            query_embedding: 512-D normalized vector
            query_face_img: Aligned face crop
        """
        img = cv2.imread(query_image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {query_image_path}")

        # Detect and extract features (includes alignment internally)
        faces = self.app.get(img)

        if len(faces) == 0:
            raise ValueError("No face detected in query image")

        if len(faces) > 1:
            print(f"Warning: {len(faces)} faces found, using the largest one")
            faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)

        query_face = faces[0]
        query_embedding = query_face.normed_embedding  # Already L2-normalized 512-D

        # Extract aligned face crop
        bbox = query_face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        query_face_img = img[y1:y2, x1:x2]

        print(f"Query embedding extracted: {query_embedding.shape}")
        return query_embedding, query_face_img

    def verify_faces(self, embedding1, embedding2, threshold=0.6):
        """
        ArcFace verification: Determine if two faces belong to same person

        Args:
            embedding1, embedding2: Normalized 512-D embeddings
            threshold: Verification threshold (default 0.6 for ArcFace)

        Returns:
            is_match: Boolean indicating if same person
            similarity: Similarity score
        """
        similarity = np.dot(embedding1, embedding2)
        is_match = similarity >= threshold
        return is_match, similarity
    def detect_faces(self, frame, confidence_threshold, verification_threshold,
                    query_embedding, timestamp, match_count, frame_count, output_folder):
        """
        Detect and verify faces in a single frame

        Args:
            frame: Video frame
            confidence_threshold: Face detection confidence threshold
            verification_threshold: Face verification threshold
            query_embedding: Query person's face embedding
            timestamp: Frame timestamp in seconds
            match_count: Current match counter
            frame_count: Current frame number
            output_folder: Folder to save matched faces

        Returns:
            match_info: Dict with match information or None
        """
        try:
            # Step 1-3: Detect → Align → Embed (all done by InsightFace)
            faces = self.app.get(frame)

            for face_idx, face in enumerate(faces):
                # Check detection confidence
                if face.det_score < confidence_threshold:
                    continue

                # Step 4: ArcFace Verification
                face_embedding = face.normed_embedding
                is_match, similarity = self.verify_faces(
                    query_embedding,
                    face_embedding,
                    threshold=verification_threshold
                )

                # Check if this is our target person (verified by ArcFace)
                if is_match:
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox

                    # Extract face crop
                    face_crop = frame[y1:y2, x1:x2]

                    # Save matched face with metadata
                    timestamp_str = str(timedelta(seconds=timestamp)).replace(':', '-')
                    filename = f"match_{match_count:04d}_frame{frame_count:06d}_{timestamp_str}_sim{similarity:.3f}.jpg"
                    filepath = os.path.join(output_folder, filename)
                    cv2.imwrite(filepath, face_crop)

                    # Store match info
                    match_info = {
                        'frame_number': frame_count,
                        'timestamp': timestamp,
                        'timestamp_formatted': str(timedelta(seconds=timestamp)),
                        'bbox': {'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)},
                        'similarity': float(similarity),
                        'confidence': float(face.det_score),
                        'verification': 'MATCH',  # ArcFace verified match
                        'filename': filename
                    }
                    return match_info

            return None

        except Exception as e:
            print(f"Error processing frame {frame_count}: {str(e)}")
            return None

    def process_video_and_find_person(self,
                                      video_path,
                                      query_embedding,
                                      output_folder,
                                      similarity_threshold=0.6,
                                      verification_threshold=0.6,
                                      frame_skip=1,
                                      confidence_threshold=0.5,
                                      save_full_frame=False):
        """
        Process video to find query person using ArcFace verification
        Args:
            video_path: Path to video
            query_embedding: 512-D embedding from query image
            output_folder: Folder to save matched faces
            similarity_threshold: (deprecated, use verification_threshold)
            verification_threshold: ArcFace verification threshold (0.6 = strict, 0.4 = lenient)
            frame_skip: Process every nth frame
            confidence_threshold: Face detection confidence
            save_full_frame: If True, save full frames with detections for debugging

        Returns:
            matches: List of dicts with timestamp, bbox, similarity, frame_num
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

        # Store video information for Phase 2
        video_info = {
            'fps': float(fps),
            'total_frames': total_frames,
            'width': width,
            'height': height,
            'duration': float(duration),
            'video_path': video_path,
            'frame_skip': frame_skip,
            'verification_threshold': verification_threshold,
            'confidence_threshold': confidence_threshold
        }

        print(f"\n{'='*60}")
        print(f"VIDEO INFO")
        print(f"{'='*60}")
        print(f"Total frames: {total_frames}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps:.2f}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Processing every {frame_skip} frame(s)")
        print(f"Verification threshold: {verification_threshold}")
        print(f"{'='*60}\n")

        matches = []
        frame_count = 0
        processed_count = 0
        match_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames if needed
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            processed_count += 1
            timestamp = frame_count / fps

            # Detect and verify faces
            match_info = self.detect_faces(
                frame,
                confidence_threshold,
                verification_threshold,
                query_embedding,
                timestamp,
                match_count,
                frame_count,
                output_folder
            )

            if match_info:
                matches.append(match_info)
                match_count += 1

            # Progress update
            if processed_count % 50 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% | Frames: {frame_count}/{total_frames} | Matches: {match_count}")

            frame_count += 1

        cap.release()

        # Save metadata in NEW FORMAT with video_info wrapper
        metadata = {
            'video_info': video_info,
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
        print(f"EXTRACTION COMPLETE")
        print(f"{'='*60}")
        print(f"Total frames processed: {processed_count}")
        print(f"Total matches found: {match_count}")
        print(f"Match rate: {match_count/processed_count*100:.2f}%" if processed_count > 0 else "N/A")
        print(f"Output folder: {output_folder}")
        print(f"Metadata saved: {metadata_path}")
        print(f"{'='*60}\n")

        return matches

