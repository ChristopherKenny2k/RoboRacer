# RoboRacer 
### (Christopher Kenny & Scott Carey)
RoboRacer is a collaborative deep learning project aimed at developing a model for a self-driving remote control car. 
The training dataset consists of 1192 reference photos taken by the camera on the remote control car.

By using the PyTorch machine learning library in python we trained three models on the performance of this dataset. 

## Model 1: Simple Convolutional Neural Network 
The CNN class defines a convolutional neural network architecture using PyTorch for image
classification. The goal of the CNN is to accurately predict the X and Y coordinates for the racing line when given the input image.

### Model Architecture
  - Convolutional Layers: Two convolutional layers with ReLU activation.
  - Pooling Layers: Max pooling with a 2x2 kernel.
  - Fully Connected Layers: Two fully connected layers for classification.
  - Perofrmance: Our simple CNN yielded a well balanced validation loss and training loss.
                 Training loss - 0.0017
                 Validation loss - 0.002

  During training, model parameters are optimized using an optimizer and a loss function, and backpropagation updates parameters to minimize the loss.

## Model 2: Transfer Learning (pre-trained CNN model)
  - Pre-Trained Model: We decided to use the ResNet18 model as our pre-trained model. Initially we chose the ResNet50 model, however, this model was heavily     
    overfit on our training data. The weights of the pre-trained model were frozen to prevent them fro mbeing updated during training.
  - ResNet18: ResNet18 is a convolutional neural network architecture introduced by Microsoft Research, consisting of 18 layers, primarily known for its 
    effectiveness in image classification tasks.
  - Dropout Rate: We used a dropout rate of 0.35 to handle some overfitting that was occuring with the ResNet18 model.
  - Performance: In comparison to our simple CNN, the loss values of the ResNet18 model fluctuated heavily.

## Model 2: Transfer Learning (pre-trained CNN model)

