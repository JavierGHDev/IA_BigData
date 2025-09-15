import os
import random
import numpy as np

# Image manipulation and GUI
import tkinter as tk  
from tkinter import messagebox
from PIL import Image, ImageTk

# tensorflow
from keras.datasets import cifar10
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical

# Model file name
MODEL_FILE = 'cifar10_model.h5'
# Folder to save images
IMAGE_FOLDER = "Cifar10_Images"

# CIFAR10 classes
classes = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "boat",
    9: "truck"
}

# Create image folder if it doesn't exist, or clear it if it does
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)
else:
    for file in os.listdir(IMAGE_FOLDER):
        file_path = os.path.join(IMAGE_FOLDER, file)
        if os.path.isfile(file_path):
            os.remove(file_path)    

# Load images for training and testing
(training_images, training_labels), (test_images, test_labels) = cifar10.load_data()

# Pick 10 random images from the test set
picked_indexes = random.sample(range(len(test_images)), 10)

# Picked images names
picked_images = []

for i, idx in enumerate(picked_indexes):
    # Convert array to image using PIL
    img = Image.fromarray(test_images[idx])
    file_name = f'img{i}_{classes[test_labels[idx][0]]}.png'
    file_path = os.path.join(IMAGE_FOLDER, file_name)
    img.save(file_path)
    picked_images.append(file_path)


# Check whether to train the model or load an existing one
if not os.path.exists(MODEL_FILE):
    print("Model not found. Training a new model. This may take a while...")
    
    # Neural network arquitecture: https://www.tensorflow.org/guide/keras/sequential_model
    model = Sequential()
    # First convolutional layer: 32 filters of 3x3 with ReLU activation
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3))) # CIFAR-10 images are 32x32 pixels with 3 color channels
    # Pooling layer to reduce spatial dimensions (gets regions of 2x2, that is 4 pixels with their corresponding values, and reduces them to 1 pixel with the maximum value of the 4)
    model.add(MaxPooling2D((2,2)))
    # Second convolutional layer
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    # Flatten the 3D output to 1D for the dense layers. At this point we work with the features extracted without regard for spatial position.
    model.add(Flatten())
    # Dense layer with 64 neurons and ReLU activation
    model.add(Dense(64, activation='relu'))
    # Exit layer with 10 neurons (number of classes, each returns  the probability of the tested image beeing the corresponding class) and softmax activation
    model.add(Dense(10, activation='softmax'))
    
    # Complete CNN flow so far:
    # 1. Conv2D + ReLU → detect local patterns
    # 2. MaxPooling → reduce spatial dimensions
    # 3. Flatten → convert 3D features → 1D vector
    # 4. Dense + ReLU → combine features
    # 5. Dense + Softmax → classify into 10 classes

    # Model compilation using Adam optimizer and categorical loss function
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Preprocessing training images
    training_images_norm = training_images.astype('float32') / 255.0
    # Formatting the labels
    training_labels_cat = to_categorical(training_labels, 10)
    
    # Define a subset of images to speed up training (for demonstration purposes)
    subset_length = 5000
    model.fit(training_images_norm[:subset_length], training_labels_cat[:subset_length],
               epochs=1, batch_size=64, verbose=1)
    
    # Save the trained model to an h5 file
    model.save(MODEL_FILE)
    print("Model has been trained and saved.")
else:
    # Load existing model
    model = load_model(MODEL_FILE)
    print("Model has been loaded.")

# Tinker GUI

# Main window
window = tk.Tk()
window.title("CIFAR-10 Image Classifier") 
window.geometry("800x600")

picked_image = tk.StringVar()

# Selected image by default
picked_image.set(picked_images[0])

# Label for dropdown menu
title_label = tk.Label(window, text="Pick an image from the folder 'Cifar10_Images'", font=("Arial", 14))
title_label.pack(pady=10)

# Image picking dropdown menu
menu_imagenes = tk.OptionMenu(window, picked_image, *picked_images)
menu_imagenes.pack(pady=10)

# Label for each image
image_label = tk.Label(window)
image_label.pack(pady=10)

# Label for the prediction result
result_label = tk.Label(window, text="", font=("Arial", 12))
result_label.pack(pady=10)

def predict_imagen():
    """
    Function for:
      - Load the selected image from the dropdown menu.
      - Show the selected image in the GUI.
      - Preprocess the image for prediction and classify it using the trained model.
      - Show the prediction result in the GUI.
    """

    file_path = picked_image.get()
    
    # Open the image using PIL
    try:
        image = Image.open(file_path)
    except Exception as e:
        messagebox.showerror("Error", f"Unable to open image: {e}")
        return
    
    # Image resizing (300x300 píxeles)
    resized_image = image.resize((300,300), Image.NEAREST)
    image_tk = ImageTk.PhotoImage(resized_image)
    image_label.config(image=image_tk)
    image_label.image = image_tk  # Keep a reference to avoid garbage collection
    
    # Image processing for prediction
    # CIFAR-10 images are 32x32 pixels
    image_pred = image.resize((32,32), Image.NEAREST)
    # Convert image to a numpy array
    image_array = np.array(image_pred)
    # Pixel value scaling to range: [0,1] --> Neural networks perform better with small input values that are in a short range
    image_array = image_array.astype('float32') / 255.0
    # Add a new dimension to match the input shape of the Keras model: (batch_size, height, width, channels)
    # original shape: (32,32,3) --> new shape: (1,32,32,3)
    image_array = np.expand_dims(image_array, axis=0) # axis=0 adds a new dimension at the start of the array shape
    
    # Prediction
    prediction = model.predict(image_array)
    # Get the index of the class with the highest predicted probability
    predicted_indx = np.argmax(prediction)
    # Get the confidence percentage of the prediction
    confidence = prediction[0][predicted_indx] * 100
    
    # Get the class name from the index
    class_name = classes.get(predicted_indx, "Unknown")
    
    # Update the result label with the prediction
    result_label.config(text=f"Actual: {picked_image.get()}, Prediction: {class_name}, Confidence: {confidence:.2f}%")

# Button to trigger the prediction
prediction_button = tk.Button(window, text="Classify image", command=predict_imagen)
prediction_button.pack(pady=10)

# Start the Tkinter event loop
window.mainloop()