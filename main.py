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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(contrast_enhanced, None, h=30, templateWindowSize=7, searchWindowSize=21)
    blurred = cv2.GaussianBlur(denoised, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((2, 2), np.uint8)
    morph_open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return morph_open

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
        
        frame = cv2.resize(frame, (640, 480))
        preprocessed_frame = preprocess_image(frame)
        boxes = perform_ocr(preprocessed_frame)

        confidences = np.array(boxes['conf'])
        
        # Check if there are valid confidence values
        if np.any(confidences > 0):
            dynamic_threshold = np.percentile(confidences[confidences > 0], 75)  # 75th percentile of non-zero confidences
        else:
            dynamic_threshold = 0  # Fallback threshold if no valid scores are found
        
        detected_text = ""
        n_boxes = len(boxes['level'])
        for i in range(n_boxes):
            if int(boxes['conf'][i]) > dynamic_threshold:  # Use dynamic threshold
                (x, y, w, h) = (boxes['left'][i], boxes['top'][i], boxes['width'][i], boxes['height'][i])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), 2)
                cv2.rectangle(preprocessed_frame, (x, y), (x + w, y + h), (50, 50, 50), 2)
                detected_text += boxes['text'][i] + " "

        if detected_text.strip():
            print("Detected Text:", detected_text.strip())
            cv2.putText(frame, detected_text.strip(), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)

        cv2.imshow("Original Live Feed", frame)
        cv2.imshow("Preprocessed for OCR", preprocessed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
