import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from PIL import Image
import numpy as np

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the images to values between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Create and compile the model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images into 1D arrays
    layers.Dense(128, activation='relu'),  # Fully connected layer with 128 neurons
    layers.Dropout(0.2),                   # Dropout for regularization
    layers.Dense(10, activation='softmax') # Output layer with 10 classes (digits 0-9)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')

# File selection dialog
root = Tk()
root.withdraw()  # Hide the main Tkinter window
print("Please select an image file.")
file_path = askopenfilename(title="Select an Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])

if file_path:
    # Open and preprocess the selected image
    image = Image.open(file_path)
    image = image.resize((28, 28))  # Resize to 28x28

    # Check if the image is grayscale, if not, add an extra dimension
    if len(image.mode) != 'L':
      image = image.convert('L')  # Convert to grayscale if needed
    image_array = np.array(image) / 255.0  # Normalize the image

    # Reshape the image array to have a batch dimension of 1
    image_array = image_array.reshape(1, 28, 28)  # Reshape to match the input shape

    # Make a prediction
    prediction = model.predict(image_array)
    predicted_label = prediction.argmax()

    # Display the image and the predicted label
    plt.imshow(image_array[0], cmap=plt.cm.binary)
    plt.title(f"Predicted: {predicted_label}")
    plt.show()

    print(f'Predicted label for the selected image: {predicted_label}')
else:
    print("No file selected. Exiting.")
