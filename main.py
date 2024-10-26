import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Uncomment and adjust path if needed

def perform_ocr(image):
    custom_config = r'--oem 3 --psm 6'
    boxes = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
    return boxes

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

        boxes = perform_ocr(frame)
        detected_text = ""
        n_boxes = len(boxes['level'])
        for i in range(n_boxes):
            if boxes['conf'][i] > 0:
                (x, y, w, h) = (boxes['left'][i], boxes['top'][i], boxes['width'][i], boxes['height'][i])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                detected_text += boxes['text'][i] + " "

        if detected_text.strip():
            print("Detected Text:", detected_text.strip())

        cv2.imshow("Live Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
