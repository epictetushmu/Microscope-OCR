import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button, Toplevel, Scale, HORIZONTAL
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ImageHistogramViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Viewer and Histogram")

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
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_pil.thumbnail((400, 400))
        self.photo = ImageTk.PhotoImage(image_pil)
        self.image_label.config(image=self.photo)
        self.image_label.image = self.photo

    def show_histogram(self):
        if self.gray_image is None:
            return

        hist_window = Toplevel(self.root)
        hist_window.title("Image Histogram")
        hist_window.geometry("600x600")

        fig = plt.Figure(figsize=(5, 3), dpi=100)
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
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=False)

        stats_frame = tk.Frame(hist_window)
        stats_frame.pack(fill=tk.X, padx=10, pady=5)

        min_val = np.min(self.gray_image)
        max_val = np.max(self.gray_image)
        mean_val = np.mean(self.gray_image)
        std_val = np.std(self.gray_image)
        median_val = np.median(self.gray_image)

        stats_text = f"Min: {min_val:.1f}  Max: {max_val:.1f}  Mean: {mean_val:.1f}  Std Dev: {std_val:.1f}  Median: {median_val:.1f}"
        Label(stats_frame, text=stats_text).pack()

        # Slider frame
        slider_frame = tk.Frame(hist_window)
        slider_frame.pack(pady=10)

        Label(slider_frame, text="Enter Intensity Center:").pack(side=tk.LEFT, padx=(0, 5))
        self.intensity_entry = tk.Entry(slider_frame, width=5)
        self.intensity_entry.insert(0, str(int(mean_val)))
        self.intensity_entry.pack(side=tk.LEFT)

        # Render button
        render_button = Button(hist_window, text="Render Selected Area", command=self.render_selected_range)
        render_button.pack(pady=10)
        
        # Find and display histogram peak
        peak_idx = np.argmax(hist)
        peak_value = int(hist[peak_idx])
        peak_text = f"Histogram Peak: Intensity {peak_idx} (Count: {peak_value})"
        peak_label = Label(hist_window, text=peak_text)
        peak_label.pack(pady=5)

    def render_selected_range(self):
        if self.gray_image is None:
            return

        try:
            center = int(self.intensity_entry.get())
            # Ensure center is in valid range
            center = max(0, min(255, center))
        except ValueError:
            # Default to mean if invalid input
            center = int(np.mean(self.gray_image))
            
        lower = max(0, center - 12)
        upper = min(255, center + 12)

        # Create mask and apply it
        mask = cv2.inRange(self.gray_image, lower, upper)
        filtered_image = cv2.bitwise_and(self.gray_image, self.gray_image, mask=mask)

        # Convert to color image for display
        display_img = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2RGB)

        # Show in new window
        result_window = Toplevel(self.root)
        result_window.title(f"Filtered Image: {lower}–{upper}")

        image_pil = Image.fromarray(display_img)
        image_pil.thumbnail((400, 400))
        tk_image = ImageTk.PhotoImage(image_pil)

        label = Label(result_window, image=tk_image)
        label.image = tk_image  # keep a reference
        label.pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageHistogramViewer(root)
    root.mainloop()
