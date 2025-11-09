
import cv2
import json
import os
from pathlib import Path
from datetime import timedelta, datetime
from collections import defaultdict
import numpy as np


class InvestigationReportGenerator:
    """
    Generate easy-to-understand reports for investigation officers and normal people
    Outputs:
    1. Simple text summary
    2. HTML visual report
    3. Video clips of sightings
    4. Excel-compatible CSV
    5. Timeline visualization
    """

    def __init__(self):
        self.report_data = {}

    def generate_complete_report(self, tracking_results, video_path, query_image_path,
                                 output_folder, case_name="Missing Person Case"):
        """
        Generate all report formats

        Args:
            tracking_results: Results from tracking pipeline
            video_path: Original video file path
            query_image_path: Query person image
            output_folder: Where to save reports
            case_name: Name of the investigation case
        """
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"GENERATING INVESTIGATION REPORTS")
        print(f"{'='*60}\n")

        # Load video info
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        video_name = Path(video_path).name

        # Process tracking results
        sightings = self._process_sightings(tracking_results, fps)

        # 1. Generate simple text summary
        print("✓ Generating text summary...")
        self._generate_text_summary(
            sightings, video_name, duration, query_image_path,
            output_folder / "INVESTIGATION_SUMMARY.txt", case_name
        )

        # 2. Generate CSV for Excel
        print("✓ Generating CSV report...")
        self._generate_csv_report(
            sightings, video_name,
            output_folder / "sightings_report.csv"
        )

        # 3. Generate HTML visual report
        print("✓ Generating HTML report...")
        self._generate_html_report(
            sightings, video_name, duration, query_image_path,
            output_folder / "visual_report.html", case_name
        )

        # 4. Extract video clips of sightings
        print("✓ Extracting video clips...")
        clips_folder = output_folder / "video_clips"
        self._extract_sighting_clips(
            video_path, sightings, clips_folder
        )

        # 5. Create timeline image
        print("✓ Creating timeline visualization...")
        self._create_timeline_image(
            sightings, duration,
            output_folder / "timeline.png"
        )

        print(f"\n{'='*60}")
        print(f"REPORTS GENERATED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"Location: {output_folder}")
        print(f"Files:")
        print(f"  📄 INVESTIGATION_SUMMARY.txt - Quick overview")
        print(f"  📊 sightings_report.csv - Excel-compatible data")
        print(f"  🌐 visual_report.html - Visual report (open in browser)")
        print(f"  📈 timeline.png - Visual timeline")
        print(f"  🎬 video_clips/ - Individual sighting clips")
        print(f"{'='*60}\n")

        return str(output_folder / "INVESTIGATION_SUMMARY.txt")

    def _process_sightings(self, tracking_results, fps):
        """Group continuous sightings into segments"""
        if not tracking_results:
            return []

        sightings = []
        current_segment = None
        gap_threshold = 2.0  # seconds - if gap > 2s, new segment

        for result in tracking_results:
            timestamp = result['timestamp']

            if current_segment is None:
                # Start new segment
                current_segment = {
                    'start_time': timestamp,
                    'end_time': timestamp,
                    'start_frame': result['frame_number'],
                    'end_frame': result['frame_number'],
                    'confidence_scores': [result.get('reid_similarity', 0)],
                    'face_verified': result.get('face_verified', False),
                    'track_id': result.get('track_id', 'N/A')
                }
            else:
                # Check if this continues the segment
                if timestamp - current_segment['end_time'] <= gap_threshold:
                    # Continue segment
                    current_segment['end_time'] = timestamp
                    current_segment['end_frame'] = result['frame_number']
                    current_segment['confidence_scores'].append(
                        result.get('reid_similarity', 0)
                    )
                    if result.get('face_verified', False):
                        current_segment['face_verified'] = True
                else:
                    # Save current segment and start new one
                    current_segment['duration'] = (
                        current_segment['end_time'] - current_segment['start_time']
                    )
                    current_segment['avg_confidence'] = np.mean(
                        current_segment['confidence_scores']
                    )
                    sightings.append(current_segment)

                    # Start new segment
                    current_segment = {
                        'start_time': timestamp,
                        'end_time': timestamp,
                        'start_frame': result['frame_number'],
                        'end_frame': result['frame_number'],
                        'confidence_scores': [result.get('reid_similarity', 0)],
                        'face_verified': result.get('face_verified', False),
                        'track_id': result.get('track_id', 'N/A')
                    }

        # Add last segment
        if current_segment:
            current_segment['duration'] = (
                current_segment['end_time'] - current_segment['start_time']
            )
            current_segment['avg_confidence'] = np.mean(
                current_segment['confidence_scores']
            )
            sightings.append(current_segment)

        return sightings

    def _format_timestamp(self, seconds):
        """Convert seconds to HH:MM:SS format"""
        return str(timedelta(seconds=int(seconds)))

    def _generate_text_summary(self, sightings, video_name, duration,
                               query_image_path, output_path, case_name):
        """Generate simple text summary for quick reading"""

        with open(output_path, 'w') as f:
            # Header
            f.write("="*70 + "\n")
            f.write(f"INVESTIGATION REPORT: {case_name}\n".center(70))
            f.write("="*70 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Query Image: {Path(query_image_path).name}\n")
            f.write(f"Video File: {video_name}\n")
            f.write(f"Video Duration: {self._format_timestamp(duration)}\n\n")

            # Quick Summary
            f.write("="*70 + "\n")
            f.write("QUICK SUMMARY\n")
            f.write("="*70 + "\n")

            if not sightings:
                f.write("\n❌ NO SIGHTINGS FOUND\n")
                f.write("The person was not detected in this video footage.\n\n")
            else:
                total_duration = sum(s['duration'] for s in sightings)
                face_verified_count = sum(1 for s in sightings if s['face_verified'])

                f.write(f"\n✓ PERSON DETECTED: YES\n")
                f.write(f"✓ Total Sightings: {len(sightings)}\n")
                f.write(f"✓ Total Time Visible: {self._format_timestamp(total_duration)}\n")
                f.write(f"✓ Face Verified Sightings: {face_verified_count}/{len(sightings)}\n\n")

                # Detailed Sightings
                f.write("="*70 + "\n")
                f.write("DETAILED SIGHTINGS\n")
                f.write("="*70 + "\n\n")

                for idx, sighting in enumerate(sightings, 1):
                    f.write(f"SIGHTING #{idx}\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"  Time Range: {self._format_timestamp(sighting['start_time'])} ")
                    f.write(f"to {self._format_timestamp(sighting['end_time'])}\n")
                    f.write(f"  Duration: {sighting['duration']:.1f} seconds\n")
                    f.write(f"  Confidence: {sighting['avg_confidence']*100:.1f}%\n")
                    f.write(f"  Face Verified: {'✓ YES' if sighting['face_verified'] else '✗ NO (body match only)'}\n")
                    f.write(f"  Track ID: {sighting['track_id']}\n")
                    f.write(f"  Video Clip: video_clips/sighting_{idx:02d}.mp4\n")
                    f.write("\n")

                # Key Moments (highest confidence sightings)
                f.write("="*70 + "\n")
                f.write("KEY MOMENTS (HIGHEST CONFIDENCE)\n")
                f.write("="*70 + "\n\n")

                sorted_sightings = sorted(sightings,
                                         key=lambda x: x['avg_confidence'],
                                         reverse=True)

                for idx, sighting in enumerate(sorted_sightings[:3], 1):
                    f.write(f"{idx}. At {self._format_timestamp(sighting['start_time'])}")
                    f.write(f" - Confidence: {sighting['avg_confidence']*100:.1f}%")
                    if sighting['face_verified']:
                        f.write(" [FACE VERIFIED]")
                    f.write("\n")

            # Footer
            f.write("\n" + "="*70 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*70 + "\n")

    def _generate_csv_report(self, sightings, video_name, output_path):
        """Generate CSV file that can be opened in Excel"""

        with open(output_path, 'w') as f:
            # Header
            f.write("Sighting #,Video File,Start Time,End Time,Duration (sec),")
            f.write("Confidence %,Face Verified,Track ID,Video Clip\n")

            # Data rows
            for idx, sighting in enumerate(sightings, 1):
                f.write(f"{idx},")
                f.write(f"{video_name},")
                f.write(f"{self._format_timestamp(sighting['start_time'])},")
                f.write(f"{self._format_timestamp(sighting['end_time'])},")
                f.write(f"{sighting['duration']:.1f},")
                f.write(f"{sighting['avg_confidence']*100:.1f},")
                f.write(f"{'Yes' if sighting['face_verified'] else 'No'},")
                f.write(f"{sighting['track_id']},")
                f.write(f"video_clips/sighting_{idx:02d}.mp4\n")

    def _generate_html_report(self, sightings, video_name, duration,
                             query_image_path, output_path, case_name):
        """Generate visual HTML report"""

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Investigation Report - {case_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 32px;
        }}
        .summary-box {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 36px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .stat-label {{
            font-size: 14px;
            opacity: 0.9;
        }}
        .sighting {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            border-left: 5px solid #667eea;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .sighting h3 {{
            margin-top: 0;
            color: #667eea;
        }}
        .sighting-details {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .detail-item {{
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
        }}
        .detail-label {{
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
        }}
        .detail-value {{
            font-size: 16px;
            font-weight: bold;
            color: #333;
        }}
        .verified {{
            background: #d4edda;
            color: #155724;
            padding: 5px 10px;
            border-radius: 5px;
            font-weight: bold;
        }}
        .not-verified {{
            background: #fff3cd;
            color: #856404;
            padding: 5px 10px;
            border-radius: 5px;
        }}
        .timeline {{
            margin: 20px 0;
            text-align: center;
        }}
        .timeline img {{
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .no-sightings {{
            background: #f8d7da;
            color: #721c24;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            font-size: 18px;
        }}
        .confidence-bar {{
            width: 100%;
            height: 20px;
            background: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 10px;
        }}
        .confidence-fill {{
            height: 100%;
            background: linear-gradient(90deg, #28a745, #20c997);
            transition: width 0.3s;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Investigation Report</h1>
        <h2>{case_name}</h2>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="summary-box">
        <h2>📹 Video Information</h2>
        <p><strong>File:</strong> {video_name}</p>
        <p><strong>Duration:</strong> {self._format_timestamp(duration)}</p>
    </div>
"""

        if not sightings:
            html += """
    <div class="no-sightings">
        <h2>❌ No Sightings Found</h2>
        <p>The person was not detected in this video footage.</p>
    </div>
"""
        else:
            total_duration = sum(s['duration'] for s in sightings)
            face_verified = sum(1 for s in sightings if s['face_verified'])
            avg_confidence = np.mean([s['avg_confidence'] for s in sightings]) * 100

            html += f"""
    <div class="summary-box">
        <h2>📊 Detection Summary</h2>
        <div class="summary-stats">
            <div class="stat-card">
                <div class="stat-label">Total Sightings</div>
                <div class="stat-value">{len(sightings)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Time Visible</div>
                <div class="stat-value">{total_duration:.0f}s</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Face Verified</div>
                <div class="stat-value">{face_verified}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Confidence</div>
                <div class="stat-value">{avg_confidence:.0f}%</div>
            </div>
        </div>
    </div>

    <div class="timeline">
        <h2>📈 Visual Timeline</h2>
        <img src="timeline.png" alt="Detection Timeline">
    </div>

    <div class="summary-box">
        <h2>🎯 Detailed Sightings</h2>
"""

            for idx, sighting in enumerate(sightings, 1):
                verified_badge = ('<span class="verified">✓ FACE VERIFIED</span>'
                                 if sighting['face_verified']
                                 else '<span class="not-verified">BODY MATCH ONLY</span>')

                html += f"""
        <div class="sighting">
            <h3>Sighting #{idx} {verified_badge}</h3>
            <div class="sighting-details">
                <div class="detail-item">
                    <div class="detail-label">Start Time</div>
                    <div class="detail-value">{self._format_timestamp(sighting['start_time'])}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">End Time</div>
                    <div class="detail-value">{self._format_timestamp(sighting['end_time'])}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Duration</div>
                    <div class="detail-value">{sighting['duration']:.1f}s</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Track ID</div>
                    <div class="detail-value">{sighting['track_id']}</div>
                </div>
            </div>
            <div style="margin-top: 15px;">
                <div class="detail-label">Confidence Level</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {sighting['avg_confidence']*100:.0f}%"></div>
                </div>
                <div style="text-align: right; margin-top: 5px; font-weight: bold;">
                    {sighting['avg_confidence']*100:.1f}%
                </div>
            </div>
            <p style="margin-top: 15px;">
                <strong>📹 Video Clip:</strong>
                <code>video_clips/sighting_{idx:02d}.mp4</code>
            </p>
        </div>
"""

            html += """
    </div>
"""

        html += """
</body>
</html>
"""

        with open(output_path, 'w') as f:
            f.write(html)

    def _extract_sighting_clips(self, video_path, sightings, output_folder):
        """Extract individual video clips for each sighting"""
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        for idx, sighting in enumerate(sightings, 1):
            # Add 2 seconds buffer before and after
            start_frame = max(0, sighting['start_frame'] - int(2 * fps))
            end_frame = sighting['end_frame'] + int(2 * fps)

            output_path = output_folder / f"sighting_{idx:02d}.mp4"

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            # Extract frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            for frame_num in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break

                # Add timestamp overlay
                timestamp = frame_num / fps
                time_str = self._format_timestamp(timestamp)
                cv2.putText(frame, f"Time: {time_str}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Sighting #{idx}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                out.write(frame)

            out.release()

        cap.release()

    def _create_timeline_image(self, sightings, duration, output_path):
        """Create visual timeline of sightings"""
        # Create image
        width = 1200
        height = 300
        img = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Draw timeline bar
        timeline_y = height // 2
        timeline_start_x = 50
        timeline_end_x = width - 50
        timeline_width = timeline_end_x - timeline_start_x

        # Background bar
        cv2.rectangle(img, (timeline_start_x, timeline_y - 20),
                     (timeline_end_x, timeline_y + 20), (200, 200, 200), -1)

        # Time markers
        for i in range(0, 11):
            x = timeline_start_x + int(timeline_width * i / 10)
            time_val = duration * i / 10
            time_str = self._format_timestamp(time_val)

            cv2.line(img, (x, timeline_y - 25), (x, timeline_y + 25), (100, 100, 100), 2)
            cv2.putText(img, time_str, (x - 30, timeline_y + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Plot sightings
        for sighting in sightings:
            start_x = timeline_start_x + int((sighting['start_time'] / duration) * timeline_width)
            end_x = timeline_start_x + int((sighting['end_time'] / duration) * timeline_width)

            color = (0, 200, 0) if sighting['face_verified'] else (0, 150, 255)

            cv2.rectangle(img, (start_x, timeline_y - 15),
                         (end_x, timeline_y + 15), color, -1)

        # Title
        cv2.putText(img, "Detection Timeline", (width // 2 - 100, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Legend
        cv2.rectangle(img, (50, height - 80), (70, height - 60), (0, 200, 0), -1)
        cv2.putText(img, "Face Verified", (80, height - 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.rectangle(img, (50, height - 50), (70, height - 30), (0, 150, 255), -1)
        cv2.putText(img, "Body Match Only", (80, height - 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.imwrite(str(output_path), img)
