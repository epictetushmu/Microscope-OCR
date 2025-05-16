import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button, Toplevel
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ObjectDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dual Rectangle Detector")

        # UI Elements
        self.label = Label(root, text="Select an Image")
        self.label.pack()

        self.select_button = Button(root, text="Select Image", command=self.load_image)
        self.select_button.pack()

        self.detect_button = Button(root, text="Detect", command=self.detect_objects, state=tk.DISABLED)
        self.detect_button.pack()
        
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
            self.detect_button.config(state=tk.NORMAL)
            self.histogram_button.config(state=tk.NORMAL)
    
    def display_image(self, image):
        """Display image in main window."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_pil.thumbnail((400, 400))
        self.photo = ImageTk.PhotoImage(image_pil)
        self.image_label.config(image=self.photo)

    def show_histogram(self):
        """Display grayscale histogram in a new window."""
        if self.gray_image is None:
            return
            
        # Create a new window for the histogram
        hist_window = Toplevel(self.root)
        hist_window.title("Image Histogram")
        hist_window.geometry("500x400")
        
        # Create figure and plot
        fig = plt.Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        # Calculate histogram
        hist = cv2.calcHist([self.gray_image], [0], None, [256], [0, 256])
        
        # Plot histogram
        ax.plot(hist, color='black')
        ax.set_xlim([0, 256])
        ax.set_title('Grayscale Histogram')
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Add the plot to the window
        canvas = FigureCanvasTkAgg(fig, master=hist_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add some histogram statistics
        stats_frame = tk.Frame(hist_window)
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Calculate statistics
        min_val = np.min(self.gray_image)
        max_val = np.max(self.gray_image)
        mean_val = np.mean(self.gray_image)
        std_val = np.std(self.gray_image)
        median_val = np.median(self.gray_image)
        
        # Display statistics
        stats_text = f"Min: {min_val:.1f}  Max: {max_val:.1f}  Mean: {mean_val:.1f}  Std Dev: {std_val:.1f}  Median: {median_val:.1f}"
        stats_label = Label(stats_frame, text=stats_text)
        stats_label.pack()

    def detect_objects(self):
        if self.cv_image is None:
            return

        # Preprocessing steps
        gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = np.ones((5, 5), np.uint8)
        closed_edges = cv2.dilate(edges, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = self.cv_image.copy()
        cv2.drawContours(contour_img, contours, -1, (255, 0, 0), 2)

        merged_contour_img = self.cv_image.copy()
        verification_img = self.cv_image.copy()  # New image for verification
        
        largest_contour = None
        if contours:
            # Use the largest contour (by area) for detection.
            largest_contour = max(contours, key=cv2.contourArea)
            # Compute the outer (minimum area) rectangle from the largest contour.
            outer_rect = cv2.minAreaRect(largest_contour)
            outer_box = cv2.boxPoints(outer_rect)
            outer_box = np.int32(outer_box)
            cv2.drawContours(merged_contour_img, [outer_box], -1, (255, 0, 0), 2)

            # Find the largest rectangle (red) completely inside the largest contour.
            inner_rect, verification_points = self.largest_fitting_quad(outer_rect, largest_contour)
            if inner_rect is not None:
                box = cv2.boxPoints(inner_rect)
                box = np.int32(box)
                cv2.drawContours(merged_contour_img, [box], -1, (0, 0, 255), 2)
                
                # Create verification image
                cv2.drawContours(verification_img, [largest_contour], -1, (255, 0, 0), 2)  # Draw contour
                cv2.drawContours(verification_img, [box], -1, (0, 0, 255), 2)  # Draw inner box
                
                # Mark all verification points
                for i, point in enumerate(verification_points):
                    point_type, status, coordinates = point
                    color = (0, 255, 0) if status else (0, 0, 255)  # Green if inside, red if outside
                    cv2.circle(verification_img, tuple(map(int, coordinates)), 3, color, -1)
                    # Add small text label
                    cv2.putText(verification_img, point_type, tuple(map(int, [coordinates[0]+5, coordinates[1]+5])), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        self.show_processing_steps(self.cv_image, gray, edges, closed_edges, contour_img, merged_contour_img, verification_img)

    def largest_fitting_quad(self, outer_rect, contour):
        """
        Given an outer rectangle (from cv2.minAreaRect) and the contour,
        this function searches for the largest quadrilateral that is completely contained 
        in the contour. Each corner can have independent adjustments while maintaining a valid shape.
        Returns a rotated rectangle in OpenCV format and verification points.
        """
        (cx, cy), (ow, oh), angle = outer_rect
        best_area = 0
        best_candidate = ((cx, cy), (0, 0), angle)  # Default minimal candidate
        best_verification_points = []

        # Get the corners of the original rectangle
        original_box = np.array(cv2.boxPoints(outer_rect))

        # Fewer steps for better performance
        steps = 11  # Try scale factors from 0.0 to 1.0 in 11 steps (resolution of 0.1)

        def check_quad_inside(quad, num_samples=10):
            # Collect verification points for visualization
            verification_points = []
            
            # Check if all vertices are inside the contour
            for i, point in enumerate(quad):
                inside = cv2.pointPolygonTest(contour, tuple(point), False) > 0
                verification_points.append((f"V{i}", inside, point))
                if not inside:
                    return False, verification_points

            # Check points along each edge
            for i in range(4):
                p1 = quad[i]
                p2 = quad[(i+1) % 4]
                for j, t in enumerate(np.linspace(0, 1, num_samples)):
                    sample_point = p1 * (1 - t) + p2 * t
                    inside = cv2.pointPolygonTest(contour, tuple(sample_point), False) > 0
                    verification_points.append((f"E{i}_{j}", inside, sample_point))
                    if not inside:
                        return False, verification_points

            return True, verification_points

        # Instead of a 4D search, we'll approach this differently
        # We'll search for width and height scale independently, and then adjust corners individually
        for sx in np.linspace(0.3, 1.0, steps):  # Start from 0.3 to avoid tiny rectangles
            for sy in np.linspace(0.3, 1.0, steps):
                # Create base scaled rectangle
                scaled_rect = ((cx, cy), (ow * sx, oh * sy), angle)
                scaled_box = np.array(cv2.boxPoints(scaled_rect))

                # Now try adjusting corners with smaller deviations
                for c1 in np.linspace(0.9, 1.0, 3):  # Corner 1 scale
                    for c2 in np.linspace(0.9, 1.0, 3):  # Corner 2 scale
                        for c3 in np.linspace(0.9, 1.0, 3):  # Corner 3 scale
                            for c4 in np.linspace(0.9, 1.0, 3):  # Corner 4 scale
                                # Apply adjustments to each corner
                                adjusted_quad = scaled_box.copy()
                                corner_scales = [c1, c2, c3, c4]

                                for i in range(4):
                                    vec = scaled_box[i] - np.array([cx, cy])
                                    adjusted_quad[i] = np.array([cx, cy]) + vec * corner_scales[i]

                                inside, verification_points = check_quad_inside(adjusted_quad)
                                if inside:
                                    # Convert back to rotated rectangle format
                                    # This ensures we return a valid OpenCV rotated rectangle
                                    rect = cv2.minAreaRect(np.float32(adjusted_quad))
                                    area = rect[1][0] * rect[1][1]

                                    if area > best_area:
                                        best_area = area
                                        best_candidate = rect
                                        best_verification_points = verification_points

        return best_candidate, best_verification_points

    def show_processing_steps(self, orig_img, gray, edges, closed_edges, contour_img, merged_contour_img, verification_img=None):
        """Displays processing steps in a grid layout."""
        proc_window = Toplevel(self.root)
        proc_window.title("Processing Steps")

        processing_steps = [
            ("Original Image", cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)),
            ("Grayscale", gray),
            ("Edge Detection", edges),
            ("Closed Edges", closed_edges),
            ("Contours", cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB)),
            ("Result", cv2.cvtColor(merged_contour_img, cv2.COLOR_BGR2RGB))
        ]
        
        # Add verification image if available
        if verification_img is not None:
            processing_steps.append(("Verification", cv2.cvtColor(verification_img, cv2.COLOR_BGR2RGB)))

        tk_images = []
        cols = 3  # We'll use 3 columns for layout
        
        for i, (text, img) in enumerate(processing_steps):
            row = i // cols
            col = i % cols
            
            pil_img = Image.fromarray(img)
            pil_img.thumbnail((250, 250))
            tk_img = ImageTk.PhotoImage(pil_img)
            tk_images.append(tk_img)

            Label(proc_window, text=text).grid(row=row*2, column=col, padx=5, pady=2)
            Label(proc_window, image=tk_img).grid(row=row*2+1, column=col, padx=5, pady=2)

        proc_window.tk_images = tk_images

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectorApp(root)
    root.mainloop()