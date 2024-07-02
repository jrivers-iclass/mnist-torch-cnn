from torch import nn

# Image Classifier ConvNet
class ImageClassifier(nn.Module): 
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)),  # Input layer with 1 channel, 32 filters, and 3x3 kernel
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)), # Hidden layer with 32 filters, 64 filters, and 3x3 kernel
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)), # Hidden layer with 64 filters, 64 filters, and 3x3 kernel
            nn.ReLU(),
            nn.Flatten(), # Reshape the output to 1D tensor
            nn.Linear(64*(28-6)*(28-6), 10)  # Output layer with 10 classes, 64*(28-6)*(28-6) input features (loses 2 pixels per conv layer)
        )

    def forward(self, x): 
        return self.model(x)