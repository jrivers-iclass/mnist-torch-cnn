    # Import dependencies
import torch 
from torch import load
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from ImageClassifier import ImageClassifier

# Hyperparameters
epochs = 10 # Number of times to iterate over the dataset
learning_rate = 0.001 # Learning rate for the optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Get test data 
test = datasets.MNIST(root="data", download=True, train=False, transform=ToTensor())
dataset = DataLoader(test, 32) # Batch size of 32, 60000 images (mini batches)


# Instance of the neural network
model = ImageClassifier().to(device)

# Load the existing model weights
with open('model_state.pt', 'rb') as f: 
        model.load_state_dict(load(f))  

# Test the model
correct = 0
total = 0
with torch.no_grad():
    for images, labels in dataset:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        _, predicted = torch.max(output, 1)
        total += labels.shape[0]
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct/total
    print(f"Accuracy: {accuracy}%")