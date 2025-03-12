import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pandas as pd

# Constants
IMAGE_FOLDER = "images"      # Folder where your images are stored
CSV_FILE = "labels.csv"      # CSV file created by your labeling tool
IMG_HEIGHT, IMG_WIDTH = 128, 128  # Target dimensions for your images

# Load the model
model = load_model('chip_detector_model.h5')

# Load CSV data to map chip names to labels
df = pd.read_csv(CSV_FILE)
chip_names = df['chip_name'].unique()
label_to_index = {name: idx for idx, name in enumerate(chip_names)}

# Function to load and preprocess an image
def load_image(image_path):
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img)
    img_array /= 255.0  # Normalize pixel values to [0, 1]
    return img_array

# Function to get the label from the filename
def get_label_from_filename(filename):
    row = df[df['filename'] == filename]
    if not row.empty:
        return label_to_index[row['chip_name'].iloc[0]]
    return None

# Function to handle image selection and prediction
def select_image():
    # Open a file dialog to let the user select an image file
    image_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
    )
    
    if not image_path:  # If the user cancels the file selection
        return

    # Load the selected image and make a prediction
    image = load_image(image_path)
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Predict the image class
    prediction = model.predict(image)
    predicted_class = np.round(prediction[0][0])  # Round to either 0 or 1 for binary classification

    # Get the true label from the filename
    true_label = get_label_from_filename(os.path.basename(image_path))

    if true_label is None:
        messagebox.showerror("Error", f"Label for {os.path.basename(image_path)} not found in CSV.")
        return

    # Display results in a message box
    if predicted_class == true_label:
        accuracy = 100  # Correct prediction
        result_message = f"Prediction: Correct\nAccuracy: {accuracy}%"
    else:
        accuracy = 0  # Incorrect prediction
        result_message = f"Prediction: Incorrect\nAccuracy: {accuracy}%"

    messagebox.showinfo("Prediction Result", f"Image: {os.path.basename(image_path)}\n{result_message}")


# Creating the main application window
def create_app():
    app = tk.Tk()
    app.title("Chip Detection")

    # Add a button to select an image and make a prediction
    predict_button = tk.Button(app, text="Select Image and Predict", command=select_image)
    predict_button.pack(pady=20)

    # Start the GUI loop
    app.mainloop()


if __name__ == "__main__":
    create_app()
