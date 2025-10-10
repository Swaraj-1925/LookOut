import cv2
import torchvision.transforms as T
import json
import os
import numpy as np
from  feature_extractor import FeatureExtractor

class ReIDPipeline:
    def __init__(self, use_osnet=False):
        print("\n" + "="*60)
        print("INITIALIZING RE-ID PIPELINE")
        print("="*60 + "\n")

        self.osnet = FeatureExtractor(use_osnet)

        # For backward compatibility with basic features
        self.transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print("Re-ID Pipeline initialized\n")

    def extract_reid_features(self, body_image):
        if body_image is None or body_image.size == 0:
            return None

        features = {}

        deep_features = self.osnet.extract_features(body_image)
        if deep_features is not None:
            features['osnet_features'] = deep_features.tolist()

        # Extract color histogram (secondary - for lighting variations)
        hsv = cv2.cvtColor(body_image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                            [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features['color_histogram'] = hist.tolist()

        # Extract spatial features (tertiary - for body structure)
        img_resized = cv2.resize(body_image, (64, 128))
        h, w = img_resized.shape[:2]
        regions = [
            img_resized[0:h//3, :],
            img_resized[h//3:2*h//3, :],
            img_resized[2*h//3:h, :]
        ]
        spatial_feat = []
        for region in regions:
            mean_color = cv2.mean(region)[:3]
            spatial_feat.extend(mean_color)
        features['spatial_features'] = spatial_feat

        return features

    def compute_feature_similarity(self, features1, features2, weights=None):
        if features1 is None or features2 is None:
            return 0.0

        if weights is None:
            weights = {
                'osnet_features': 0.7,      # Primary weight on deep features
                'color_histogram': 0.2,     # Secondary weight on color
                'spatial_features': 0.1     # Tertiary weight on structure
            }

        total_similarity = 0.0
        total_weight = 0.0

        for feat_name, weight in weights.items():
            if feat_name in features1 and feat_name in features2:
                feat1 = np.array(features1[feat_name])
                feat2 = np.array(features2[feat_name])

                # Cosine similarity
                sim = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-6)
                total_similarity += weight * sim
                total_weight += weight

        return total_similarity / (total_weight + 1e-6)

    def create_reid_gallery(self, body_detections_path, output_path):
        print(f"\n{'='*60}")
        print(f"CREATING RE-ID GALLERY")
        print(f"{'='*60}")

        metadata_path = os.path.join(body_detections_path, 'body_detections_metadata.json')
        if not os.path.exists(metadata_path):
            raise ValueError(f"Metadata not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            body_metadata = json.load(f)

        gallery = []

        for idx, body_info in enumerate(body_metadata):
            filename = body_info['filename']
            filepath = os.path.join(body_detections_path, filename)

            if not os.path.exists(filepath):
                continue

            body_img = cv2.imread(filepath)
            features = self.extract_reid_features(body_img)

            if features:
                gallery_entry = {
                    'id': idx,
                    'filename': filename,
                    'frame_number': body_info['frame_number'],
                    'timestamp': body_info['timestamp'],
                    'features': features,
                    'face_similarity': body_info.get('similarity', 0.0)
                }
                gallery.append(gallery_entry)

            if (idx + 1) % 50 == 0:
                print(f"Processed: {idx + 1}/{len(body_metadata)}")

        # Save gallery
        with open(output_path, 'w') as f:
            json.dump(gallery, f, indent=2)

        print(f"✓ gallery created: {len(gallery)} entries")
        print(f"✓ Saved to: {output_path}")
        print(f"{'='*60}\n")

        return gallery
