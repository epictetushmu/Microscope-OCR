import cv2
import numpy as np
import pytesseract
from spellchecker import SpellChecker
from collections import Counter

# Path to Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust path if needed

# Initialize the spell checker
spell = SpellChecker()

def perform_ocr(image):
    """Perform OCR on the preprocessed image using LSTM engine."""
    custom_config = r'--oem 3 --psm 6'  # Use OEM 3 for LSTM engine
    return pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)

def sharpen_image(image):
    """Sharpen the image to enhance text clarity."""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def correct_spelling(text):
    """Correct spelling of detected text using a spell checker."""
    words = text.split()
    corrected = []
    
    for word in words:
        # Get the best candidate for each word
        candidates = spell.candidates(word)
        # Select the first candidate if there are any candidates
        corrected_word = next(iter(candidates), word) if candidates else word
        corrected.append(corrected_word)
    
    return ' '.join(corrected)

def preprocess_image(frame):
    """Preprocess the image for better OCR results."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(gray)
    sharpened = sharpen_image(contrast_enhanced)
    denoised = cv2.fastNlMeansDenoising(sharpened, None, h=30, templateWindowSize=7, searchWindowSize=21)
    blurred = cv2.GaussianBlur(denoised, (5, 5), 0)  # Use smaller kernel for Gaussian Blur
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((2, 2), np.uint8)
    morph_open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return morph_open

def draw_boxes(frame, boxes, dynamic_threshold):
    """Draw bounding boxes around detected text and collect detected text."""
    detected_text = ""
    n_boxes = len(boxes['level'])
    
    for i in range(n_boxes):
        if int(boxes['conf'][i]) > dynamic_threshold:
            (x, y, w, h) = (boxes['left'][i], boxes['top'][i], boxes['width'][i], boxes['height'][i])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), 2)
            detected_text += boxes['text'][i] + " "
    
    return detected_text.strip()

def main():
    """Main function to capture video and perform OCR."""
    cap = cv2.VideoCapture(0)

    # Set camera resolution to a lower value for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    # Initialize a counter to store detected words
    word_counter = Counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Resize frame to reduce processing load
        frame = cv2.resize(frame, (640, 480))
        
        # Perform preprocessing and OCR
        preprocessed_frame = preprocess_image(frame)
        boxes = perform_ocr(preprocessed_frame)

        # Compute dynamic threshold
        confidences = np.array(boxes['conf'])
        dynamic_threshold = np.percentile(confidences[confidences > 0], 75) if np.any(confidences > 0) else 0
        
        detected_text = draw_boxes(frame, boxes, dynamic_threshold)

        if detected_text:
            corrected_text = correct_spelling(detected_text)  # Correct spelling
            print("Detected Text:", corrected_text)

            # Split corrected text into words and update the counter
            for word in corrected_text.split():
                word_counter[word] += 1

            # Combine the most common words to form the expected text
            expected_text = analyze_expected_words(word_counter)
            print("Expected Text:", expected_text)

            # Display the detected and expected texts on the frame
            cv2.putText(frame, f"Detected: {corrected_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
            cv2.putText(frame, f"Expected: {expected_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Original Live Feed", frame)
        cv2.imshow("Preprocessed for OCR", preprocessed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def analyze_expected_words(word_counter):
    """Analyze the word counts and find the most probable expected words."""
    most_common = word_counter.most_common(2)  # Get top 2 common words
    expected_text = []

    # Combine high-frequency words to form expected phrases
    for word, count in most_common:
        expected_text.append(word)

    return ' '.join(expected_text)

if __name__ == "__main__":
    main()
