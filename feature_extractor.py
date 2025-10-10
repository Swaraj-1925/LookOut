import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as T
import torchreid

class FeatureExtractor:
    def __init__(self, use_osnet = False, model_name='osnet_x1_0'):

        if(use_osnet):
            self.model = torchreid.models.build_model(
                name=model_name,
                num_classes=1000,
                pretrained=True
            )
            self.model.eval()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)
            self.use_osnet = True
            print(f"OSNet loaded: {model_name} on {self.device}")
        else:
            self.use_osnet = False
            # Fallback to ResNet50
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.model.fc = nn.Identity()  # Remove final classification layer
            self.model.eval()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)
            print(f"resnet50 loaded: on {self.device}")


        # Standard ReID transforms
        self.transform = T.Compose([
            T.Resize((256, 128)),  # Standard ReID resolution
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, image):
        if image is None or image.size == 0:
            return None

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        # Transform and add batch dimension
        img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # Extract features
        with torch.no_grad():
            features = self.model(img_tensor)

        # Normalize features
        features = features.cpu().numpy().flatten()
        features = features / (np.linalg.norm(features) + 1e-6)

        return features

