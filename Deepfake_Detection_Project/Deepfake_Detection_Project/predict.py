import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys

# ======================
# 1. Define CNN (same as training)
# ======================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


def main():

    if len(sys.argv) < 2:
        print("Usage: python3 predict.py image_path")
        return

    image_path = sys.argv[1]

    # ======================
    # 2. Load Model
    # ======================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    model.load_state_dict(torch.load("cnn_large_dataset.pth", map_location=device))
    model.eval()

    # ======================
    # 3. Transform Image
    # ======================
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    # ======================
    # 4. Predict
    # ======================
    with torch.no_grad():
        output = model(image)
        confidence = output.item()

    if confidence > 0.5:
        prediction = "Real"
    else:
        prediction = "Fake"

    print("\nPrediction:", prediction)
    print("Confidence:", round(confidence, 4))


if __name__ == "__main__":
    main()