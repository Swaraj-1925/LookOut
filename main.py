from look_out import LookOut
from report_generator import InvestigationReportGenerator
from pathlib import Path
import time

query_image="./dummy_dataset/q/yt_hikaru_nakamura_vid2.jpg"
video_path="./dummy_dataset/vids/yt_hikaru_nakamura_vid2.mp4"
output_dir="./test8"
case_name="John Doe - Missing Person Case #12345"
start_time = time.time()
pipeline = LookOut(use_osnet=False)
report_generator = InvestigationReportGenerator()
results = pipeline.run_complete_pipeline(
        query_image=query_image,
        video_path=video_path,
        output_dir=output_dir,
        # Threshold tuning for accuracy:
        face_threshold=0.30,      # Lower = more initial candidates (0.5-0.6)
        reid_threshold=0.60,      # Higher = fewer false positives (0.6-0.75)
        verification_threshold=0.30,
        confidence_threshold= 0.25,

        # Performance vs accuracy:
        frame_skip=3,             # 1 = best accuracy, 2-3 = faster

        # Accuracy boosters:
        use_face_verification=True,   # HIGHLY RECOMMENDED for accuracy
        use_temporal_filter=True      # Reduces false positives
        )
end_time = time.time()
elapsed_sec = end_time - start_time
if results:
    report_path = report_generator.generate_complete_report(
            tracking_results=results['tracking'],
            video_path=video_path,
            query_image_path=query_image,
            output_folder=Path(output_dir) / "investigation_reports",
            case_name=case_name
            )

    print(f"\n{'='*60}")
    print(f"INVESTIGATION SUMMARY AVAILABLE AT:")
    print(f"   {report_path}")
    print(f"{'='*60}\n")

print(f"[INFO] Total pipeline runtime: {elapsed_sec:.2f} seconds\n")
