# Import dependencies
import torch 
from torch import nn, save
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from ImageClassifier import ImageClassifier

# Hyperparameters
epochs = 10 # Number of times to iterate over the dataset
learning_rate = 0.001 # Learning rate for the optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Get data 
train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(train, 32) # Batch size of 32, 60000 images (mini batches)
#1,28,28 - classes 0-9
    
# Create the model, optimizer, and loss function    
model = ImageClassifier().to(device) # Instance of the neural network
opt = Adam(model.parameters(), lr=learning_rate) # Optimizer for updating the weights
loss_fn = nn.CrossEntropyLoss()  # Loss function for classification

# Train the model
for epoch in range(epochs):
    for i, (images, labels) in enumerate(dataset):
        images, labels = images.to(device), labels.to(device) # Move the tensors to the device
        opt.zero_grad() # Zero the gradients
        output = model(images) # Get the model's predictions
        loss = loss_fn(output, labels) # Calculate the loss
        loss.backward() # Backpropagate the loss
        opt.step() # Update the weights
        print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item()}")

# Save the model
with open('model_state.pt', 'wb') as f:
    save(model.state_dict(), f)
