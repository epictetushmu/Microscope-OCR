# Microscope OCR

Microscope OCR is a tool designed to recognize and identify electronic components through a microscope feed, providing automatic detection and essential information retrieval. The OCR functionality can be used with any microscope setup that provides a digital image feed, making it versatile for electronics hobbyists and professionals alike.

## Project Overview

This project focuses on OCR-based recognition of small electronic components under a microscope. Once detected, the tool provides useful information like datasheets, color codes, and component specs. Current functionality includes identifying resistors (via color codes), ICs, and other common components.

### Key Features
- **Universal Compatibility**: Works with any microscope setup that outputs a digital feed.
- **Component Detection**:
  - **Resistor Color Codes**: Identifies resistance values from color bands.
  - **IC Recognition**: Detects IC part numbers and searches for datasheets.
  - **Other Components**: Expanding detection for diodes, capacitors, etc.
- **Automated Datasheet Lookup**: Retrieves datasheets and specs for recognized components.

## Getting Started

### Prerequisites
- **Software Requirements**:
  - Python 3.7+
  - [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
  - Additional Python libraries: `opencv-python`, `numpy` , `time` , `BeautifulSoup` , `requests`
  
