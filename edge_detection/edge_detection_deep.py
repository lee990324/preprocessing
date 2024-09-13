import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
import torch.nn as nn

# Define the Edge Detection model
class EdgeDetectionNet(nn.Module):
    def __init__(self):
        super(EdgeDetectionNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.sigmoid(self.conv4(x))
        return x

# Load the trained model
model = EdgeDetectionNet()
model.load_state_dict(torch.load('edge_detection_model.pth'))
model.eval()

# Transformation pipeline
transform = transforms.Compose([
    transforms.ToTensor()
])

# Function to process each frame
def process_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(Variable(image))
    
    output = output.squeeze().numpy()  # Remove batch dimension and convert to numpy
    output = (output * 255).astype(np.uint8)  # Convert to 8-bit image
    output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)  # Convert to BGR for saving
    return output

# Open the video file
input_video_path = 'C:/Users/gram15/Desktop/Working_dir/data/video3_.mp4'
output_video_path = 'C:/Users/gram15/Desktop/Working_dir/data/video3_deep.mp4'

cap = cv2.VideoCapture(input_video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    processed_frame = process_frame(frame)
    out.write(processed_frame)

cap.release()
out.release()
cv2.destroyAllWindows()

print('Edge detection completed and saved to', output_video_path)