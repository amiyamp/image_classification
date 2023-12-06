# Celebrity Image Classification
## Project Overview
This project focuses on building a deep learning model to classify images of five celebrities: Lionel Messi, Maria Sharapova, Roger Federer, Serena Williams, and Virat Kohli.

## Chosen Model
Employed a Convolutional Neural Network (CNN) architecture for image classification. The model consists of convolutional layers, max-pooling layers, and fully connected layers. The final layer uses softmax activation to output probabilities for each celebrity class.

### Model Architecture
- Input Shape: (128, 128, 3)
- Convolutional Layers: 32 filters with a (3,3) kernel and ReLU activation
- Max Pooling: (2,2)
- Flatten Layer
- Dense Layers: 256 neurons with ReLU activation, Dropout (50%), 512 neurons with ReLU activation
- Output Layer: 5 neurons with softmax activation

## Training Process
The dataset is divided into training and testing sets. The images are resized to (128, 128), normalized, and split using `train_test_split`. The model is compiled with the Adam optimizer and sparse categorical crossentropy loss. The training process involves 50 epochs with a batch size of 32.

### Normalization
Images in the training and testing sets are normalized by scaling pixel values to the range [0, 1].

### Training Summary
Training is conducted with a validation split of 30%. The training progress is monitored using the `history` object. Training accuracy and loss are visualized over epochs.

## Critical Findings
Insights gained during the training process include model accuracy, loss, and validation metrics. A classification report is generated to evaluate the model's performance on the test set.

## Model Evaluation
After training, the model is evaluated on the test set, yielding an accuracy of approximately 80%. The classification report provides a detailed breakdown of precision, recall, and F1-score for each class.

## Model Prediction
A sample image (`serena_williams7.png`) is provided for prediction. The model predicts the class, and the result is printed in the console.



