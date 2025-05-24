import tkinter as tk
from tkinter import filedialog, Label, Button, Text
from PIL import Image, ImageTk
import pytesseract

# Set path to tesseract executable (adjust as needed)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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
        # Open the image and perform OCR directly without preprocessing
        img = Image.open(img_path)
        
        # Extract text using pytesseract on the original image
        extracted_text = pytesseract.image_to_string(img).strip()
        
        # Display the OCR results
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, f"Detected text:\n{extracted_text}")

# Create the main window
root = tk.Tk()
root.title("Simple OCR - No Preprocessing")

img_path = None
original_image_pil = None

# Button to select image
select_btn = Button(root, text="Select Image", command=select_image)
select_btn.pack(pady=10)

# Label to display original image preview
preview_label = Label(root)
preview_label.pack(pady=10)

# Button to process image - runs OCR without effects
process_btn = Button(root, text="Run OCR", command=process_image)
process_btn.pack(pady=10)

# Text widget to display OCR result
result_text = Text(root, height=10, width=50)
result_text.pack(pady=10)

# Run the GUI event loop
root.mainloop()
