# Import dependencies
import torch 
from PIL import Image
from torch import nn, load
from torch.optim import Adam
from torchvision.transforms import ToTensor
from ImageClassifier import ImageClassifier

# Hyperparameters
epochs = 10
learning_rate = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Instance of the neural network, loss, optimizer 
model = ImageClassifier().to(device)
opt = Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss() 

with open('model_state.pt', 'rb') as f: 
        model.load_state_dict(load(f))  

img = Image.open('test_images/img_1.jpg') 

img_tensor = ToTensor()(img).unsqueeze(0).to(device)

print(torch.argmax(model(img_tensor)))