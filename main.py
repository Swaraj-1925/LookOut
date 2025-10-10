from look_out import LookOut


pipeline = LookOut(use_osnet=False)
results = pipeline.run_complete_pipeline(
    query_image="/home/kevin/Projects/LookOut/dataset/q/yt_speed_vid1.jpg",
    video_path="/home/kevin/Projects/LookOut/dataset/vids/yt_speed_vid1.mp4",
    output_dir="./test1",

    # Threshold tuning for accuracy:
    face_threshold=0.55,      # Lower = more initial candidates (0.5-0.6)
    reid_threshold=0.65,      # Higher = fewer false positives (0.6-0.75)

    # Performance vs accuracy:
    frame_skip=1,             # 1 = best accuracy, 2-3 = faster

    # Accuracy boosters:
    use_face_verification=True,   # HIGHLY RECOMMENDED for accuracy
    use_temporal_filter=True      # Reduces false positives
)

