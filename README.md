# Getting Started
- Create a new venv with `python -m venv .venv`
- Activate the venv with `source ./.venv/bin/activate`

# Install Dependencies
- `pip install -r requirements.txt`

# Train the model
This will take some time and output a model in the root directory `model_state.pt`
- `python train.py`

# Test the model
You will find a line that looks like:
```img = Image.open('test_images/img_1.jpg') ```
Just update the image to test the model against different images. 
The available images are: 
- img_1.jpg (Should be a 2)
- img_2.jpg (Should be a 0)
- img_3.jpg (should be a 9)

To run the inferencing use:
- `python inference.py`


# Documentation
- [Pytorch](https://pytorch.org/)
- [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
- [Back Propagation](https://en.wikipedia.org/wiki/Backpropagation)
- [ConvNet](https://en.wikipedia.org/wiki/Convolutional_neural_network)


# Cuda
- install [conda](https://github.com/conda-forge/miniforge)
- install [cuda](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)