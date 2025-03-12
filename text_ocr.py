import tkinter as tk
from tkinter import filedialog, Label, Button, Text, Toplevel
from PIL import Image, ImageTk, ImageOps, ImageDraw
import cv2
import pytesseract
import numpy as np

# Set path to tesseract executable (adjust as needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    # Convert image to grayscale for better OCR performance
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to enhance text regions (parameters may be adjusted)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    return image, gray, thresh

def detect_text_regions(gray_image):
    # Use pytesseract to get bounding boxes for text regions.
    data = pytesseract.image_to_data(gray_image, output_type=pytesseract.Output.DICT)
    boxes = []
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        # Filter out weak detections (adjust confidence threshold as needed)
        try:
            conf = int(data['conf'][i])
        except ValueError:
            conf = 0
        if conf > 0 and data['text'][i].strip() != "":
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            boxes.append((x, y, w, h, data['text'][i]))
    return boxes

def draw_boxes_on_image(image, boxes):
    # Convert the OpenCV image (BGR) to RGB and then to a PIL image.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_img)
    for (x, y, w, h, word) in boxes:
        draw.rectangle([(x, y), (x + w, y + h)], outline="red", width=2)
    return pil_img

def select_image():
    global img_path, original_image_pil
    # Open a file dialog to select an image file
    img_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
    )
    if img_path:
        # Open the image using PIL for preview
        original_image_pil = Image.open(img_path)
        original_image_pil.thumbnail((300, 300))
        img_preview = ImageTk.PhotoImage(original_image_pil)
        preview_label.config(image=img_preview)
        preview_label.image = img_preview  # keep a reference
        result_text.delete("1.0", tk.END)  # Clear previous OCR result

def process_image():
    if img_path:
        # Process image with OpenCV: get original, grayscale, and thresholded images.
        orig_cv, gray, thresh = preprocess_image(img_path)
        
        # Detect text regions from the grayscale image
        boxes = detect_text_regions(gray)
        
        # Create an annotated image with bounding boxes drawn over the detected text areas.
        annotated_img = draw_boxes_on_image(orig_cv, boxes)
        
        # Convert processed images to PIL for display and resize them to fit the preview window.
        gray_pil = Image.fromarray(gray)
        gray_pil.thumbnail((300, 300))
        thresh_pil = Image.fromarray(thresh)
        # Convert thresholded image to RGB for proper display in Tkinter.
        thresh_pil = thresh_pil.convert("RGB")
        thresh_pil.thumbnail((300, 300))
        annotated_img.thumbnail((300, 300))
        
        # Extract text using pytesseract on the grayscale image for OCR result.
        extracted_text = pytesseract.image_to_string(gray).strip()
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, f"Detected text:\n{extracted_text}")
        
        # Open a new Toplevel window to display all intermediate images in a 2x2 grid.
        proc_window = Toplevel(root)
        proc_window.title("Processed Images - Step by Step")
        
        # Create labels for each image with their titles.
        # Original Image
        Label(proc_window, text="Original Image").grid(row=0, column=0, padx=10, pady=10)
        orig_img_tk = ImageTk.PhotoImage(original_image_pil)
        orig_label = Label(proc_window, image=orig_img_tk)
        orig_label.image = orig_img_tk  # keep reference
        orig_label.grid(row=1, column=0, padx=10, pady=10)
        
        # Grayscale Image
        Label(proc_window, text="Grayscale Image").grid(row=0, column=1, padx=10, pady=10)
        gray_img_tk = ImageTk.PhotoImage(gray_pil)
        gray_label = Label(proc_window, image=gray_img_tk)
        gray_label.image = gray_img_tk
        gray_label.grid(row=1, column=1, padx=10, pady=10)
        
        # Thresholded Image
        Label(proc_window, text="Thresholded Image").grid(row=2, column=0, padx=10, pady=10)
        thresh_img_tk = ImageTk.PhotoImage(thresh_pil)
        thresh_label = Label(proc_window, image=thresh_img_tk)
        thresh_label.image = thresh_img_tk
        thresh_label.grid(row=3, column=0, padx=10, pady=10)
        
        # Annotated Image with bounding boxes for text regions
        Label(proc_window, text="Annotated Image").grid(row=2, column=1, padx=10, pady=10)
        annotated_img_tk = ImageTk.PhotoImage(annotated_img)
        annotated_label = Label(proc_window, image=annotated_img_tk)
        annotated_label.image = annotated_img_tk
        annotated_label.grid(row=3, column=1, padx=10, pady=10)

# Create the main window
root = tk.Tk()
root.title("IC Chip OCR - Step by Step Preview")

img_path = None
original_image_pil = None

# Button to select image
select_btn = Button(root, text="Select Image", command=select_image)
select_btn.pack(pady=10)

# Label to display original image preview
preview_label = Label(root)
preview_label.pack(pady=10)

# Button to process image and show intermediate steps
process_btn = Button(root, text="Process Image", command=process_image)
process_btn.pack(pady=10)

# Text widget to display OCR result
result_text = Text(root, height=10, width=50)
result_text.pack(pady=10)

# Run the GUI event loop
root.mainloop()
