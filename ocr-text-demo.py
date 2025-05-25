import tkinter as tk
from tkinter import filedialog, Label, Button, Text, Toplevel
from PIL import Image, ImageTk, ImageOps, ImageDraw
import cv2
import pytesseract
import numpy as np

# Set path to tesseract executable (adjust as needed)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to remove noise while preserving edges
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    
    # Apply morphological operations to clean up the image
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    return image, gray, enhanced, opening

def detect_text_regions(gray_image):
    # Configure Tesseract for better chip text recognition
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    # Use pytesseract to get bounding boxes for text regions
    data = pytesseract.image_to_data(gray_image, config=custom_config, output_type=pytesseract.Output.DICT)
    
    boxes = []
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        try:
            conf = int(data['conf'][i])
        except ValueError:
            conf = 0
        if conf > 30 and data['text'][i].strip() != "":  # Increased confidence threshold
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            boxes.append((x, y, w, h, data['text'][i]))
    return boxes

def draw_boxes_on_image(image, boxes):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_img)
    for (x, y, w, h, word) in boxes:
        draw.rectangle([(x, y), (x + w, y + h)], outline="red", width=2)
        # Add text label above the box
        draw.text((x, y-15), word, fill="red")
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
        # Process image with enhanced preprocessing
        orig_cv, gray, enhanced, opening = preprocess_image(img_path)
        
        # Detect text regions from the enhanced image
        boxes = detect_text_regions(enhanced)
        
        # Create an annotated image with bounding boxes
        annotated_img = draw_boxes_on_image(orig_cv, boxes)
        
        # Convert processed images to PIL for display
        gray_pil = Image.fromarray(gray)
        gray_pil.thumbnail((300, 300))
        enhanced_pil = Image.fromarray(enhanced)
        enhanced_pil.thumbnail((300, 300))
        opening_pil = Image.fromarray(opening)
        opening_pil = opening_pil.convert("RGB")
        opening_pil.thumbnail((300, 300))
        annotated_img.thumbnail((300, 300))
        
        # Extract text using enhanced configuration
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        extracted_text = pytesseract.image_to_string(enhanced, config=custom_config).strip()
        
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, f"Detected text:\n{extracted_text}")
        
        # Display all processed images
        proc_window = Toplevel(root)
        proc_window.title("Processed Images - Step by Step")
        
        # Original Image
        Label(proc_window, text="Original Image").grid(row=0, column=0, padx=10, pady=10)
        orig_img_tk = ImageTk.PhotoImage(original_image_pil)
        orig_label = Label(proc_window, image=orig_img_tk)
        orig_label.image = orig_img_tk
        orig_label.grid(row=1, column=0, padx=10, pady=10)
        
        # Grayscale Image
        Label(proc_window, text="Grayscale Image").grid(row=0, column=1, padx=10, pady=10)
        gray_img_tk = ImageTk.PhotoImage(gray_pil)
        gray_label = Label(proc_window, image=gray_img_tk)
        gray_label.image = gray_img_tk
        gray_label.grid(row=1, column=1, padx=10, pady=10)
        
        # Enhanced Image
        Label(proc_window, text="Enhanced Image").grid(row=2, column=0, padx=10, pady=10)
        enhanced_img_tk = ImageTk.PhotoImage(enhanced_pil)
        enhanced_label = Label(proc_window, image=enhanced_img_tk)
        enhanced_label.image = enhanced_img_tk
        enhanced_label.grid(row=3, column=0, padx=10, pady=10)
        
        # Annotated Image
        Label(proc_window, text="Annotated Image").grid(row=2, column=1, padx=10, pady=10)
        annotated_img_tk = ImageTk.PhotoImage(annotated_img)
        annotated_label = Label(proc_window, image=annotated_img_tk)
        annotated_label.image = annotated_img_tk
        annotated_label.grid(row=3, column=1, padx=10, pady=10)

# Create the main window
root = tk.Tk()
root.title("Enhanced IC Chip OCR")

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
