# app.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
import logging
import time

# ----- Logging setup -----
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ----- Model definition -----
class ImprovedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Load trained weights -----
model = ImprovedCNN().to(device)
model.load_state_dict(torch.load("cnn_weights.pth", map_location=device))
model.eval()

# ----- Preprocessing -----
transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
])

def predict(img):
    start = time.time()
    try:
        img = Image.fromarray(img)  # numpy → PIL
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            label = classes[predicted.item()]
        latency = time.time() - start
        logging.info(f"Prediction: {label}, Latency={latency:.3f}s")
        return label
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return "Error in prediction!"

# Gradio app
demo = gr.Interface(fn=predict, inputs="image", outputs="label")
demo.launch(server_name="0.0.0.0", server_port=7860)

