import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button, Toplevel, Scale, HORIZONTAL, Frame, IntVar, Scrollbar, Canvas
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ScrollableFrame(tk.Frame):
    """A scrollable frame class that can automatically create scrollbars when content exceeds the frame size"""
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.canvas = Canvas(self)
        scrollbar_y = Scrollbar(self, orient="vertical", command=self.canvas.yview)
        scrollbar_x = Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.scrollable_frame = Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas_frame = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        self.canvas.bind("<Configure>", self._resize_canvas_window)
        
        self.canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        # Layout with grid to enable both scrollbars
        self.canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar_y.grid(row=0, column=1, sticky="ns")
        scrollbar_x.grid(row=1, column=0, sticky="ew")
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
    
    def _resize_canvas_window(self, event):
        """Resize the canvas window when the canvas is resized"""
        visible_width = event.width
        # Update the width of the canvas window to match visible area 
        # (prevents horizontal scrollbar when not needed)
        self.canvas.itemconfig(self.canvas_frame, width=visible_width)

class ImageHistogramViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Black/Gray Detection Tool")

        Label(root, text="Select an Image").pack()

        Button(root, text="Select Image", command=self.load_image).pack()
        self.detect_button = Button(root, text="Detect Black/Gray", command=self.show_detection_panel, state=tk.DISABLED)
        self.detect_button.pack()

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
            self.detect_button.config(state=tk.NORMAL)

    def display_image(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_pil.thumbnail((400, 400))
        self.photo = ImageTk.PhotoImage(image_pil)
        self.image_label.config(image=self.photo)
        self.image_label.image = self.photo

    def show_detection_panel(self):
        """Show the black/gray detection control panel"""
        if self.cv_image is None:
            return

        detection_window = Toplevel(self.root)
        detection_window.title("Black/Gray Detection Controls")
        detection_window.geometry("600x500")
        
        # Create scrollable frame for the controls
        main_frame = ScrollableFrame(detection_window)
        main_frame.pack(fill="both", expand=True)
        content_frame = main_frame.scrollable_frame
        
        # Configure the content frame to use available width
        content_frame.columnconfigure(0, weight=1)
        
        # Show grayscale histogram to help with threshold selection
        fig = plt.Figure(figsize=(5, 3), dpi=100)
        ax = fig.add_subplot(111)
        
        # Calculate grayscale histogram
        hist = cv2.calcHist([self.gray_image], [0], None, [256], [0, 256])
        ax.plot(hist, color='black')
        ax.set_xlim([0, 256])
        ax.set_title('Grayscale Histogram')
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Mark typical darkness threshold with a vertical line
        ax.axvline(x=50, color='red', linestyle='--', alpha=0.7, label='Default Darkness Threshold')
        ax.legend()

        fig.tight_layout()
        
        # Use the content frame for the canvas
        canvas = FigureCanvasTkAgg(fig, master=content_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Show grayscale statistics to help with threshold selection
        stats_frame = Frame(content_frame)
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        min_val = np.min(self.gray_image)
        max_val = np.max(self.gray_image)
        mean_val = np.mean(self.gray_image)
        std_val = np.std(self.gray_image)
        median_val = np.median(self.gray_image)
        
        stats_text = f"Grayscale Statistics: Min: {min_val:.1f}  Max: {max_val:.1f}  Mean: {mean_val:.1f}  Std Dev: {std_val:.1f}  Median: {median_val:.1f}"
        Label(stats_frame, text=stats_text).pack()

        # Black/Gray detection controls
        detect_frame = Frame(content_frame)
        detect_frame.pack(pady=5, fill=tk.X)
        
        Label(detect_frame, text="Black/Gray Detection Parameters:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 5))
        
        # Darkness threshold control
        darkness_frame = Frame(detect_frame)
        darkness_frame.pack(fill=tk.X, pady=5)
        Label(darkness_frame, text="Darkness Threshold (0-255):").pack(side=tk.LEFT, padx=(0, 5))
        self.darkness_var = IntVar(value=50)
        Scale(darkness_frame, from_=0, to=255, orient=HORIZONTAL, variable=self.darkness_var, length=200).pack(side=tk.LEFT)
        
        # Gray similarity threshold control
        gray_frame = Frame(detect_frame)
        gray_frame.pack(fill=tk.X, pady=5)
        Label(gray_frame, text="Color Similarity (max RGB difference):").pack(side=tk.LEFT, padx=(0, 5))
        self.similarity_var = IntVar(value=15)
        Scale(gray_frame, from_=0, to=50, orient=HORIZONTAL, variable=self.similarity_var, length=200).pack(side=tk.LEFT)
        
        # Detection mode
        mode_frame = Frame(detect_frame)
        mode_frame.pack(fill=tk.X, pady=5)
        Label(mode_frame, text="Detection Mode:").pack(side=tk.LEFT, padx=(0, 5))
        self.detect_mode = tk.StringVar(value="both")
        modes = [("Both", "both"), ("Only Black", "black"), ("Only Gray", "gray")]
        for text, value in modes:
            tk.Radiobutton(mode_frame, text=text, variable=self.detect_mode, value=value).pack(side=tk.LEFT, padx=5)
        
        # Preview option
        preview_frame = Frame(detect_frame)
        preview_frame.pack(fill=tk.X, pady=5)
        Label(preview_frame, text="Preview Style:").pack(side=tk.LEFT, padx=(0, 5))
        self.preview_mode = tk.StringVar(value="highlight")
        preview_modes = [("Highlight", "highlight"), ("Mask Only", "mask")]
        for text, value in preview_modes:
            tk.Radiobutton(preview_frame, text=text, variable=self.preview_mode, value=value).pack(side=tk.LEFT, padx=5)
        
        # Explanation of parameters
        help_frame = Frame(content_frame)
        help_frame.pack(pady=(10, 5), fill=tk.X, padx=10)
        
        help_text = """
        Parameter Guide:
        • Darkness Threshold: Pixels with all RGB values below this are considered "black"
        • Color Similarity: Maximum difference between R, G, B channels to be considered "gray"
        • Detection Mode: Choose to detect only black, only gray, or both
        • Preview Style: "Highlight" brightens detected regions, "Mask" shows only the mask
        """
        
        help_label = Label(help_frame, text=help_text, justify=tk.LEFT, wraplength=550)
        help_label.pack(anchor=tk.W)
        
        # Detect button with prominent styling
        detect_button_frame = Frame(content_frame)
        detect_button_frame.pack(pady=15)
        
        detect_button = Button(
            detect_button_frame, 
            text="Detect Black/Gray Regions", 
            command=self.detect_black_gray,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 11, "bold"),
            padx=10,
            pady=5
        )
        detect_button.pack()

    def detect_black_gray(self):
        if self.cv_image is None:
            return
        
        # Get parameters from UI
        darkness = self.darkness_var.get()
        similarity = self.similarity_var.get()
        mode = self.detect_mode.get()
        preview_style = self.preview_mode.get()
        
        # Create a copy of the original image
        result_img = np.copy(self.cv_image)
        
        # Convert to RGB for easier processing
        rgb_img = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
        
        # Create masks based on selected mode
        dark_mask = np.zeros(self.cv_image.shape[:2], dtype=np.uint8)
        gray_mask = np.zeros(self.cv_image.shape[:2], dtype=np.uint8)
        
        # Process the image to find black regions
        if mode in ["both", "black"]:
            # Black is when all channels are below darkness threshold
            dark_pixels = (rgb_img[:,:,0] < darkness) & (rgb_img[:,:,1] < darkness) & (rgb_img[:,:,2] < darkness)
            dark_mask[dark_pixels] = 255
            
        # Process the image to find gray regions
        if mode in ["both", "gray"]:
            # Calculate max difference between channels for each pixel
            r, g, b = rgb_img[:,:,0], rgb_img[:,:,1], rgb_img[:,:,2]
            max_diff = np.maximum(np.maximum(np.abs(r-g), np.abs(r-b)), np.abs(g-b))
            
            # Gray is when channels are similar (max difference < similarity threshold)
            # but not completely dark (at least one channel > darkness to avoid overlap)
            gray_pixels = (max_diff < similarity) & ~((r < darkness) & (g < darkness) & (b < darkness))
            gray_mask[gray_pixels] = 255
        
        # Combine masks based on mode
        final_mask = np.zeros(self.cv_image.shape[:2], dtype=np.uint8)
        if mode == "both":
            final_mask = cv2.bitwise_or(dark_mask, gray_mask)
        elif mode == "black":
            final_mask = dark_mask
        else:  # gray
            final_mask = gray_mask
        
        # Apply mask to original image to highlight regions
        detected_image = np.copy(self.cv_image)
        
        # Modify image based on detected regions and preview style
        if preview_style == "highlight" and np.any(final_mask):
            # Create highlighted version - make detected regions brighter 
            # Convert to HSV for brightness manipulation
            hsv = cv2.cvtColor(detected_image, cv2.COLOR_BGR2HSV)
            # Increase brightness for detected regions
            hsv[:, :, 2] = np.where(final_mask > 0, np.minimum(hsv[:, :, 2] * 1.5, 255), hsv[:, :, 2])
            detected_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        elif preview_style == "mask":
            # Show just the mask in the detected image
            detected_image = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
        
        # Calculate statistics about the detected regions
        total_pixels = self.cv_image.shape[0] * self.cv_image.shape[1]
        detected_pixels = np.count_nonzero(final_mask)
        percentage = (detected_pixels / total_pixels) * 100
        
        # Show in new window with scrolling capability
        result_window = Toplevel(self.root)
        result_window.title(f"Black/Gray Detection: {percentage:.1f}% of image")
        result_window.geometry("850x700")  # Default size
        
        # Create scrollable main frame
        scroll_frame = ScrollableFrame(result_window)
        scroll_frame.pack(fill="both", expand=True)
        content_frame = scroll_frame.scrollable_frame
        
        # Create a frame for side-by-side display
        display_frame = Frame(content_frame)
        display_frame.pack(padx=10, pady=10, fill="both")
        
        # Configure the grid to use available space
        display_frame.columnconfigure(0, weight=1)
        display_frame.columnconfigure(1, weight=1)
        
        # Show original image on the left
        original_rgb = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
        original_pil = Image.fromarray(original_rgb)
        # Allow larger thumbnails since we have scrollbars
        original_pil.thumbnail((600, 600))
        original_tk = ImageTk.PhotoImage(original_pil)
        
        Label(display_frame, text="Original Image").grid(row=0, column=0, padx=5, pady=5)
        original_label = Label(display_frame, image=original_tk)
        original_label.image = original_tk  # keep a reference
        original_label.grid(row=1, column=0, padx=5, pady=5)
        
        # Show detected regions on the right
        detected_rgb = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
        detected_pil = Image.fromarray(detected_rgb)
        detected_pil.thumbnail((600, 600))
        detected_tk = ImageTk.PhotoImage(detected_pil)
        
        detection_label_text = "Detected Regions" if preview_style == "highlight" else "Detection Mask"
        Label(display_frame, text=detection_label_text).grid(row=0, column=1, padx=5, pady=5)
        detected_label = Label(display_frame, image=detected_tk)
        detected_label.image = detected_tk  # keep a reference
        detected_label.grid(row=1, column=1, padx=5, pady=5)
        
        # Only show separate mask if in highlight mode
        if preview_style == "highlight":
            # Show mask image below
            mask_img = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2RGB)
            mask_pil = Image.fromarray(mask_img)
            mask_pil.thumbnail((800, 800))
            mask_tk = ImageTk.PhotoImage(mask_pil)
            
            Label(display_frame, text="Detection Mask").grid(row=2, column=0, columnspan=2, padx=5, pady=5)
            mask_label = Label(display_frame, image=mask_tk)
            mask_label.image = mask_tk  # keep a reference
            mask_label.grid(row=3, column=0, columnspan=2, padx=5, pady=5)
        
        # Add statistics
        stats_text = f"Detection Results:\n"
        stats_text += f"  • Mode: {mode.capitalize()}\n"
        stats_text += f"  • Darkness threshold: {darkness}\n"
        stats_text += f"  • Similarity threshold: {similarity}\n"
        stats_text += f"  • Detected pixels: {detected_pixels:,} of {total_pixels:,} ({percentage:.1f}%)"
        
        stats_label = Label(content_frame, text=stats_text, justify=tk.LEFT)
        stats_label.pack(padx=10, pady=10, anchor=tk.W)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageHistogramViewer(root)
    root.mainloop()
