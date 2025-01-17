# MNIST Handwritten Digit Recognition

This project implements a simple AI-based handwritten digit recognition system using the MNIST dataset and TensorFlow. It allows you to train a neural network, evaluate its accuracy, and make predictions on new images through a file selection dialog.

## Features

- Loads and preprocesses the MNIST dataset.
- Trains a neural network with a simple architecture.
- Provides a GUI to select and classify new images.
- Predicts handwritten digits from grayscale images.

## Prerequisites

Ensure you have the following installed:

- Python 3.7 or later
- TensorFlow 2.x
- NumPy
- Matplotlib
- Pillow
- Tkinter (usually included with Python)

## How to Run the Code

1. Clone this repository or copy the code.
2. Install the required dependencies:
   ```bash
   pip install tensorflow numpy matplotlib pillow
   ```
3. Save the code in a Python file (e.g., `main.py`).
4. Run the script:
   ```bash
   python main.py
   ```
5. After training and evaluation, a file selection dialog will open. Select an image file to classify.

## Supported Image Formats

- PNG
- JPG/JPEG
- BMP
- GIF

## Notes

- Images must be resized to 28x28 pixels and in grayscale format for accurate predictions.
- The model may not perform well on images outside the MNIST dataset distribution.

## Architecture

The neural network has the following layers:

1. **Flatten Layer**: Converts 28x28 input images into a 1D array.
2. **Dense Layer**: Fully connected layer with 128 neurons and ReLU activation.
3. **Dropout Layer**: Regularizes the model to prevent overfitting.
4. **Output Layer**: Contains 10 neurons with softmax activation for digit classification (0-9).

## Limitations

- This AI model is trained on the MNIST dataset and may not generalize well to non-MNIST images.
- Errors and misclassifications can occur, especially with noisy or non-standard digit images.

**Disclaimer**: This is an AI-based application and may have inaccuracies or issues. Always verify predictions for critical use cases.

## Example Output

1. Train the model:
   ```
   Epoch 1/5
   1875/1875 [==============================] - loss: 0.2951 - accuracy: 0.9151
   ...
   Test accuracy: 0.9748
   ```
2. Select an image to classify. The predicted digit and the image will be displayed.

## License

This project is open-source and can be used or modified for educational and research purposes.

Developer: Muhammad Salam (Git: salamalsam)
