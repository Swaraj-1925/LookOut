from collections import defaultdict
import numpy as np
from pathlib import Path
from face_recognition import FaceRecognitionPipeline
from body_detection import BodyDetectionPipeline
from reid import ReIDPipeline
from video_tracker import VideoTracker
from analysis import Analysis

class LookOut:
    def __init__(self, use_osnet=False):
        """Initialize all components"""
        print("\n" + "="*60)
        print("INITIALIZING COMPLETE PIPELINE")
        print("="*60 + "\n")

        self.face_pipeline = FaceRecognitionPipeline(
                ctx_id=0
                )
        self.body_pipeline = BodyDetectionPipeline()
        self.reid_pipeline = ReIDPipeline(use_osnet=use_osnet)
        self.analysis = Analysis()
        self.tracking_pipeline = VideoTracker(
            self.reid_pipeline,
            self.body_pipeline,
            self.face_pipeline
        )

        print("All pipeline components initialized\n")

    def run_complete_pipeline(self, query_image, video_path, output_dir,
                             face_threshold=0.6, reid_threshold=0.65, frame_skip=1,
                            verification_threshold=0.45,confidence_threshold=0.25,
                             use_face_verification=True, use_temporal_filter=True):
        """
        Run complete enhanced pipeline with ByteTrack and OSNet

        Args:
            query_image: Path to query person image
            video_path: Path to input video
            output_dir: Output directory
            face_threshold: Face matching threshold (0.5-0.7 recommended)
            reid_threshold: Re-ID threshold (0.6-0.7 for high accuracy)
            frame_skip: Process every nth frame (1 = all frames)
            use_face_verification: Enable face verification (highly recommended)
            use_temporal_filter: Enable temporal consistency filtering
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*60)
        print("STARTING PERSON RE-ID PIPELINE")
        print("="*60)
        print(f"Query: {query_image}")
        print(f"Video: {video_path}")
        print(f"Output: {output_dir}")
        print(f"Face verification: {'ON' if use_face_verification else 'OFF'}")
        print(f"Temporal filtering: {'ON' if use_temporal_filter else 'OFF'}")
        print("="*60 + "\n")

        # Phase 1: Face Recognition
        print("PHASE 1: FACE RECOGNITION (ArcFace)")
        faces_dir = output_dir / "phase1_faces"
        query_embedding, _ = self.face_pipeline.extract_query_embedding(query_image)
        matches = self.face_pipeline.process_video_and_find_person(
            video_path, query_embedding, str(faces_dir),
        verification_threshold=0.45,
        confidence_threshold=0.25,
        frame_skip=1
        )

        if not matches:
            print("No face matches found. Try lowering face_threshold.")
            return None

        print(f"✓ Found {len(matches)} face matches\n")

        # Phase 2: Body Detection
        print("PHASE 2: BODY DETECTION (YOLO + Pose)")
        bodies_dir = output_dir / "phase2_bodies"
        body_results = self.body_pipeline.process_matched_faces(
            video_path,
            str(faces_dir / "matches_metadata.json"),
            str(bodies_dir),
            strategy='hybrid'
        )

        if not body_results:
            print("No bodies detected. Exiting.")
            return None

        print(f"✓ Extracted {len(body_results)} body crops\n")

        # Phase 3: Enhanced Re-ID Gallery with OSNet
        print("PHASE 3: RE-ID GALLERY")
        gallery_path = output_dir / "reid_gallery.json"
        gallery = self.reid_pipeline.create_reid_gallery(
            str(bodies_dir),
            str(gallery_path)
        )

        print(f"✓ Gallery size: {len(gallery)} samples\n")

        # Phase 4: Video Tracking with ByteTrack
        print("PHASE 4: TRACKING (ByteTrack + Multi-Verification)")
        output_video = output_dir / "tracked_video.mp4"
        tracking_results = self.tracking_pipeline.track_person_in_video(
            video_path,
            gallery,
            str(output_video),
            query_image_path=query_image,
            query_embedding=query_embedding,
            reid_threshold=reid_threshold,
            frame_skip=frame_skip,
            use_face_verification=use_face_verification,
            use_temporal_filter=use_temporal_filter
        )

        # Generate summary report
        self._generate_summary_report(output_dir, matches, body_results, tracking_results)

        print("\n" + "="*60)
        print("ENHANCED PIPELINE COMPLETE!")
        print("="*60)
        print(f"Results: {output_dir}")
        print(f"Tracked video: {output_video}")
        print(f"Face matches: {len(matches)}")
        print(f"Body detections: {len(body_results)}")
        print(f"Tracked frames: {len(tracking_results)}")
        if tracking_results:
            detection_rate = len(tracking_results) / sum(1 for _ in Path(video_path).parent.iterdir()) * 100
            print(f"Detection rate: {len(tracking_results)} frames")
        print("="*60 + "\n")
        metadata_path = "output/phase1_faces/matches_metadata.json"
        return {
                'faces': matches,
                'bodies': body_results,
                'gallery': gallery,
                'tracking': tracking_results,
                'output_video': str(output_video)
                }

    def _generate_summary_report(self, output_dir, faces, bodies, tracking):
        """Generate a summary report of the pipeline results"""
        report_path = output_dir / "pipeline_report.txt"

        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("ENHANCED PERSON RE-IDENTIFICATION REPORT\n")
            f.write("="*60 + "\n\n")

            f.write("PHASE 1: FACE RECOGNITION\n")
            f.write(f"  - Face matches found: {len(faces)}\n")
            if faces:
                face_sims = [m['similarity'] for m in faces]
                f.write(f"  - Average similarity: {np.mean(face_sims):.3f}\n")
                f.write(f"  - Max similarity: {np.max(face_sims):.3f}\n")
                f.write(f"  - Min similarity: {np.min(face_sims):.3f}\n")
            f.write("\n")

            f.write("PHASE 2: BODY DETECTION\n")
            f.write(f"  - Bodies extracted: {len(bodies)}\n")
            if bodies:
                methods = defaultdict(int)
                for b in bodies:
                    methods[b.get('detection_method', 'unknown')] += 1
                f.write("  - Detection methods:\n")
                for method, count in methods.items():
                    f.write(f"    * {method}: {count}\n")
            f.write("\n")

            f.write("PHASE 3: RE-IDENTIFICATION\n")
            f.write("  - Features: OSNet deep features + Color + Spatial\n")
            f.write("  - Verification: Multi-level (Face + Body)\n")
            f.write("\n")

            f.write("PHASE 4: TRACKING RESULTS\n")
            f.write(f"  - Tracked frames: {len(tracking)}\n")
            if tracking:
                reid_sims = [t['reid_similarity'] for t in tracking]
                face_verified = sum(1 for t in tracking if t.get('face_verified', False))
                f.write(f"  - Average Re-ID similarity: {np.mean(reid_sims):.3f}\n")
                f.write(f"  - Face verified frames: {face_verified}/{len(tracking)}\n")
                f.write(f"  - Face verification rate: {face_verified/len(tracking)*100:.1f}%\n")
            f.write("\n")
            f.write("\n" + "="*60 + "\n")

        print(f"✓ Summary report saved: {report_path}")

