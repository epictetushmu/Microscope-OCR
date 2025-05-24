import cv2
import os
from tkinter import Tk, filedialog, Button, Label

def select_and_invert():
    file_path = filedialog.askopenfilename(
        filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
    if not file_path:
        return
    img = cv2.imread(file_path)
    if img is None:
        status_label.config(text="Failed to load image.")
        return
    inverted_img = cv2.bitwise_not(img)
    save_dir = os.path.join(os.getcwd(), "images", "Tests")
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.basename(file_path)
    name, ext = os.path.splitext(base_name)
    save_path = os.path.join(save_dir, f"{name}_inverted{ext}")
    cv2.imwrite(save_path, inverted_img)
    status_label.config(text=f"Saved: {save_path}")

root = Tk()
root.title("Invert Image Colors")

select_btn = Button(root, text="Select Image and Invert", command=select_and_invert)
select_btn.pack(pady=10)

status_label = Label(root, text="")
status_label.pack(pady=10)

root.mainloop()