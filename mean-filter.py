import cv2
import os
from tkinter import Tk, filedialog, Button, Label

def select_and_mean_filter():
    file_path = filedialog.askopenfilename(
        filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
    if not file_path:
        return
    img = cv2.imread(file_path)
    if img is None:
        status_label.config(text="Failed to load image.")
        return
    # Apply mean filter (blur)
    mean_filtered_img = cv2.blur(img, (5, 5))
    save_dir = os.path.join(os.getcwd(), "images", "Tests")
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.basename(file_path)
    name, ext = os.path.splitext(base_name)
    save_path = os.path.join(save_dir, f"{name}_meanfiltered{ext}")
    cv2.imwrite(save_path, mean_filtered_img)
    status_label.config(text=f"Saved: {save_path}")

root = Tk()
root.title("Mean Filter Image")

select_btn = Button(root, text="Select Image and Apply Mean Filter", command=select_and_mean_filter)
select_btn.pack(pady=10)

status_label = Label(root, text="")
status_label.pack(pady=10)

root.mainloop()
