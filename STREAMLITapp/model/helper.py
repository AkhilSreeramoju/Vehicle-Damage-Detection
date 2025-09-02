import ssl
print(ssl.get_default_verify_paths())

from PIL import Image
import torch
from torch import nn
from torchvision import transforms
import torchvision.models as models
import os

train_model = None
num_classes = ['Front_Breakage', 'Front_Crushed', 'Front_Normal', 'Rear_Breakage', 'Rear_Crushed', 'Rear_Normal']

class CarClassifierResnet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = models.resnet50(weights="DEFAULT")

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # in_features = self.model.classifier[1].in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x


def predict(image_path):
  image = Image.open(image_path).convert('RGB')
  transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  image_tensor = transform(image).unsqueeze_(0)
  if train_model is None:
      model = CarClassifierResnet()
      #model.load_state_dict(torch.load("model/model1.pth",map_location=torch.device("cpu")))

      model_path = os.path.join(os.path.dirname(__file__), "model1.pth")
      model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

      model.eval()
  with torch.no_grad():
      output = model(image_tensor)
      _, predicted = torch.max(output, 1)

      return num_classes[predicted.item()]

