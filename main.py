import cv2
import numpy as np
import pytesseract

# Path to Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust path if needed

def perform_ocr(image):
    custom_config = r'--oem 3 --psm 6'
    boxes = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
    return boxes

def preprocess_image(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply stronger Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)  # Increased kernel size for more blurring

    # Apply median filtering with a larger kernel to further reduce noise
    median_filtered = cv2.medianBlur(blurred, 9)  # Increased kernel size

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(median_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)

    # Optional: Morphological operations to remove small noise after thresholding
    kernel_morph = np.ones((3, 3), np.uint8)  # Structuring element for morphological operations
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_morph)  # Closing operation to fill small holes

    return morph

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Preprocess the image for better OCR results
        preprocessed_frame = preprocess_image(frame)
        boxes = perform_ocr(preprocessed_frame)
        
        detected_text = ""
        n_boxes = len(boxes['level'])
        for i in range(n_boxes):
            if boxes['conf'][i] > 0:
                (x, y, w, h) = (boxes['left'][i], boxes['top'][i], boxes['width'][i], boxes['height'][i])
                # Draw rectangles on the original frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Draw rectangles on the preprocessed frame for visualization
                cv2.rectangle(preprocessed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                detected_text += boxes['text'][i] + " "

        if detected_text.strip():
            print("Detected Text:", detected_text.strip())

        # Display both the original frame and the preprocessed frame
        cv2.imshow("Original Live Feed", frame)
        cv2.imshow("Preprocessed for OCR", preprocessed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
