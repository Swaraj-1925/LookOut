from look_out import LookOut


pipeline = LookOut(use_osnet=False)
results = pipeline.run_complete_pipeline(
    query_image="./dummy_dataset/q/yt_speed_vid1.jpg",
    video_path="./dummy_dataset/vids/yt_speed_vid1.mp4",
    output_dir="./test1",

    # Threshold tuning for accuracy:
    face_threshold=0.3,      # Lower = more initial candidates (0.5-0.6)
    reid_threshold=0.4,      # Higher = fewer false positives (0.6-0.75)

    verification_threshold=0.30,
    confidence_threshold=0.30,
    # Performance vs accuracy:
    frame_skip=1,             # 1 = best accuracy, 2-3 = faster

    # Accuracy boosters:
    use_face_verification=True,   # HIGHLY RECOMMENDED for accuracy
    use_temporal_filter=True      # Reduces false positives
)

