# Import dependencies
import torch 
from PIL import Image
from torch import load
from torchvision.transforms import ToTensor
from ImageClassifier import ImageClassifier

# Hyperparameters
epochs = 10
learning_rate = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Instance of the neural network
model = ImageClassifier().to(device)

# Load the existing model weights
with open('model_state.pt', 'rb') as f: 
        model.load_state_dict(load(f))  

# Load the test image
img = Image.open('test_images/img_1.jpg') 

# Preprocess the image
img_tensor = ToTensor()(img).unsqueeze(0).to(device)

# Make a prediction
print(torch.argmax(model(img_tensor)))