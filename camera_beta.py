import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Function to save the region inside the contour as a masked image
def save_masked_image():
    # Create a black mask of the same size as the original frame
    mask = np.zeros_like(frame)

    # Loop through each contour to fill the mask
    for contour in contours:
        # Filter out small contours (noise) by area
        if cv2.contourArea(contour) > 100:  # Adjust threshold as needed
            # Fill the contour area with white on the mask
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    # Now apply the mask to the original frame
    masked_image = cv2.bitwise_and(frame, mask)

    # Convert the masked image to RGB (OpenCV uses BGR by default)
    masked_image_rgb = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)

    # Open a save dialog to choose where to save the masked image
    filename = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
    if filename:
        cv2.imwrite(filename, masked_image)
        print(f"Masked image saved as {filename}")

# Open the camera feed
cap = cv2.VideoCapture(1)  # Use 0 for the default camera, 1 for an external camera

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

# Create a Tkinter window
root = tk.Tk()
root.title("Contour Detection with Save Masked Image Button")

# Create a button to save the masked image
save_button = tk.Button(root, text="Save Masked Image", command=save_masked_image)
save_button.pack()

# Create a canvas to display the video feed
canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

while True:
    # Capture a frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Get FPS from the camera
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Convert the frame to grayscale for line detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours based on the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original frame
    for contour in contours:
        # Filter out small contours (noise) by area
        if cv2.contourArea(contour) > 100:  # Adjust threshold as needed
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

    # Add FPS text to the frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Convert the frame to an image that Tkinter can display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img_tk = ImageTk.PhotoImage(image=img)

    # Update the image on the Tkinter canvas
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
    canvas.image = img_tk  # Keep a reference to the image to prevent garbage collection

    # Update the Tkinter window
    root.update_idletasks()
    root.update()

    # Exit when 'q' is pressed (use the keyboard to close the OpenCV window)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
root.destroy()
