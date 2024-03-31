# RoboRacer 
### (Christopher Kenny & Scott Carey)
RoboRacer is a collaborative deep learning project aimed at developing a model for a self-driving remote control car. 
The training dataset consists of 1192 reference photos taken by the camera on the remote control car.

By using the PyTorch machine learning library in python we trained three models on the performance of this dataset. 

## Model 1: Simple Convolutional Neural Network 
The CNN class defines a convolutional neural network architecture using PyTorch for image
classification.
### Model Architecture
  - Convolutional Layers: Two convolutional layers with ReLU activation.
  - Pooling Layers: Max pooling with a 2x2 kernel.
  - Fully Connected Layers: Two fully connected layers for classification.

  During training, model parameters are optimized using an optimizer and a loss function, and        backpropagation updates parameters to minimize the loss.


