# Microscope OCR

Microscope OCR is a powerful tool designed to recognize and identify electronic components through a microscope feed, providing automatic detection and essential information retrieval. This OCR functionality can be seamlessly integrated with any microscope setup that provides a digital image feed, making it versatile for both electronics hobbyists and professionals.

## Project Overview

This project focuses on OCR-based recognition of small electronic components under a microscope. Once detected, the tool provides useful information like datasheets, color codes, and component specifications. Current functionality includes identifying resistors (via color codes), integrated circuits (ICs), and other common electronic components.

### Key Features

- **Universal Compatibility**: Works with any microscope setup that outputs a digital feed.
- **Graphical User Interface (GUI)**: User-friendly main interface (`main.py`) for capturing, processing, and analyzing microscope images.
- **Component Detection**:
  - **Resistor Color Codes**: Identifies resistance values from color bands.
  - **IC Recognition**: Detects IC part numbers and searches for datasheets.
  - **Other Components**: Expanding detection capabilities for diodes, capacitors, and more.
- **Automated Datasheet Lookup**: Retrieves datasheets and specifications for recognized components (integrated with `datasheet-scraper.py`).
- **Image Preprocessing Tools**: Includes scripts for cleaning OCR input (`clear-ocr.py`), inverting image colors (`invert-color.py`), and improved black (chip) detection.
- **Histogram Analysis**: Standalone histogram tool (`histogram.py`) for image statistics and pixel intensity analysis, with integration to capture frames from the main GUI.
- **Image Export**: Save and export processed images for documentation or further analysis.
- **Customizable Area Selection**: Select area for analysis using an input text field in the GUI.

## Scripts Overview

- `main.py`: Main GUI for capturing and analyzing microscope images.
- `histogram.py`: Standalone histogram viewer and statistics tool.
- `datasheet-scraper.py`: Retrieves datasheets for detected components.
- `clear-ocr.py`: Preprocesses images to improve OCR results.
- `invert-color.py`: Inverts image colors for better contrast and detection.
- `chip-detection-demo.py`, `ocr-text-demo.py`, `shape-detection-demo.py`: Demos for specific detection and OCR tasks.

## Getting Started

### Prerequisites

To run the Microscope OCR tool, ensure you have the following software and libraries installed:

- **Software Requirements**:

  - Python 3.7 or higher
  - [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (ensure it is properly installed and configured)

- **Python Libraries**: Install the required libraries using pip:

  ```bash
  pip install opencv-python numpy BeautifulSoup4 requests
  ```

### Running the Main GUI

To start the main graphical interface, run:

```bash
python main.py
```

### Additional Tools

- To analyze an image histogram:

  ```bash
  python histogram.py
  ```

- For image preprocessing or OCR cleaning, use `clear-ocr.py` or `invert-color.py` as needed.

### Future Enhancements

- **Expanded Component Recognition**: Develop recognition capabilities for additional components like transistors, capacitors, and inductors.
- **User Interface**: Further improve the GUI for easier interaction and functionality access.
- **Improved OCR Accuracy**: Implement additional preprocessing techniques to enhance OCR accuracy in various lighting conditions.

## Contributing

Contributions are welcome! If you have suggestions for improvements or would like to add new features, please fork the repository and submit a pull request.

## License

This project is licensed under the terms of the **GNU General Public License v3.0**. This means you can freely use, modify, and distribute the software, but you must keep the same license for any derivative works.

For a copy of the license, see the [LICENSE](LICENSE) file or visit [GNU GPL v3.0](https://www.gnu.org/licenses/gpl-3.0.html).

## Acknowledgments

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for the OCR engine.
- [OpenCV](https://opencv.org/) for image processing capabilities.
- The open-source community for inspiration and collaboration.
