import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

class Analysis:
    def visualize_matches(self, output_folder, num_display=12):
        """
        Display sample matched faces in a grid

        Args:
            output_folder: Folder containing matched faces
            num_display: Number of faces to display
        """
        face_files = sorted([f for f in os.listdir(output_folder) if f.endswith('.jpg')])

        if not face_files:
            print("No matched faces found to display")
            return

        print(f"Total matched faces: {len(face_files)}")

        num_display = min(num_display, len(face_files))
        cols = 4
        rows = (num_display + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
        axes = axes.flatten() if num_display > 1 else [axes]

        for idx in range(num_display):
            face_file = face_files[idx]
            img = cv2.imread(os.path.join(output_folder, face_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Extract similarity from filename
            similarity = face_file.split('_sim')[-1].replace('.jpg', '')

            axes[idx].imshow(img)
            axes[idx].axis('off')
            axes[idx].set_title(f"Match {idx+1}\nSim: {similarity}", fontsize=10)

        # Hide unused subplots
        for idx in range(num_display, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'matched_faces_grid.jpg'), dpi=150, bbox_inches='tight')
        plt.show()

    def visualize_body_detections(self, output_folder, num_display=12):
        """
        Visualize extracted body crops

        Args:
            output_folder: Folder with body images
            num_display: Number to display
        """
        body_files = sorted([f for f in os.listdir(output_folder) if f.endswith('.jpg')])

        if not body_files:
            print("No body detections found")
            return

        print(f"Total body crops: {len(body_files)}")

        num_display = min(num_display, len(body_files))
        cols = 4
        rows = (num_display + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
        axes = axes.flatten() if num_display > 1 else [axes]

        for idx in range(num_display):
            img = cv2.imread(os.path.join(output_folder, body_files[idx]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Extract method from filename
            method = body_files[idx].split('_')[-1].replace('.jpg', '')

            axes[idx].imshow(img)
            axes[idx].axis('off')
            axes[idx].set_title(f"Body {idx+1}\nMethod: {method}", fontsize=9)

        for idx in range(num_display, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'body_detections_grid.jpg'),
                   dpi=150, bbox_inches='tight')
        plt.show()

    def get_statistics(self, matches):
        """
        Print statistics about matches
        """
        if not matches:
            print("No matches found")
            return

        similarities = [m['similarity'] for m in matches]

        print(f"\n{'='*60}")
        print(f"MATCH STATISTICS")
        print(f"{'='*60}")
        print(f"Total matches: {len(matches)}")
        print(f"Average similarity: {np.mean(similarities):.3f}")
        print(f"Max similarity: {np.max(similarities):.3f}")
        print(f"Min similarity: {np.min(similarities):.3f}")
        print(f"First appearance: {matches[0]['timestamp_formatted']}")
        print(f"Last appearance: {matches[-1]['timestamp_formatted']}")
        print(f"{'='*60}\n")
