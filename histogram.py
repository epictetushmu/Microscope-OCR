import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button, Toplevel
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ImageHistogramViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Viewer and Histogram")

        # UI Elements
        Label(root, text="Select an Image").pack()

        Button(root, text="Select Image", command=self.load_image).pack()
        self.histogram_button = Button(root, text="Show Histogram", command=self.show_histogram, state=tk.DISABLED)
        self.histogram_button.pack()

        self.image_label = Label(root)
        self.image_label.pack()

        self.image_path = None
        self.cv_image = None
        self.gray_image = None

    def load_image(self):
        self.image_path = filedialog.askopenfilename(
            filetypes=[("Image Files", ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"))]
        )
        if self.image_path:
            self.cv_image = cv2.imread(self.image_path)
            self.gray_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
            self.display_image(self.cv_image)
            self.histogram_button.config(state=tk.NORMAL)

    def display_image(self, image):
        """Display image in main window."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_pil.thumbnail((400, 400))
        self.photo = ImageTk.PhotoImage(image_pil)
        self.image_label.config(image=self.photo)
        self.image_label.image = self.photo  # Prevent garbage collection

    def show_histogram(self):
        """Display grayscale histogram in a new window."""
        if self.gray_image is None:
            return

        hist_window = Toplevel(self.root)
        hist_window.title("Image Histogram")
        hist_window.geometry("500x400")

        fig = plt.Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)

        hist = cv2.calcHist([self.gray_image], [0], None, [256], [0, 256])
        ax.plot(hist, color='black')
        ax.set_xlim([0, 256])
        ax.set_title('Grayscale Histogram')
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)

        canvas = FigureCanvasTkAgg(fig, master=hist_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        stats_frame = tk.Frame(hist_window)
        stats_frame.pack(fill=tk.X, padx=10, pady=5)

        min_val = np.min(self.gray_image)
        max_val = np.max(self.gray_image)
        mean_val = np.mean(self.gray_image)
        std_val = np.std(self.gray_image)
        median_val = np.median(self.gray_image)

        stats_text = f"Min: {min_val:.1f}  Max: {max_val:.1f}  Mean: {mean_val:.1f}  Std Dev: {std_val:.1f}  Median: {median_val:.1f}"
        Label(stats_frame, text=stats_text).pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageHistogramViewer(root)
    root.mainloop()
