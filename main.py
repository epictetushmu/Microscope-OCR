import tkinter as tk
from tkinter import Button, Label
import subprocess
import sys
import os

# Helper to run a script in a new process
def run_script(script_name):
    python_exe = sys.executable
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    subprocess.Popen([python_exe, script_path])

root = tk.Tk()
root.title("Microscope-OCR Main Launcher")
root.geometry("400x400")

Label(root, text="Microscope-OCR Toolkit", font=("Arial", 16, "bold"), pady=20).pack()

Button(root, text="Black/Gray Detection (Histogram)", width=35, pady=8,
       command=lambda: run_script("histogram.py")).pack(pady=8)
Button(root, text="OCR Text Demo", width=35, pady=8,
       command=lambda: run_script("ocr-text-demo.py")).pack(pady=8)
Button(root, text="Chip Detection Demo", width=35, pady=8,
       command=lambda: run_script("chip-detection-demo.py")).pack(pady=8)
Button(root, text="Shape Detection Demo", width=35, pady=8,
       command=lambda: run_script("shape-detection-demo.py")).pack(pady=8)
Button(root, text="Invert Color Tool", width=35, pady=8,
       command=lambda: run_script("invert-color.py")).pack(pady=8)
Button(root, text="Clear OCR Tool", width=35, pady=8,
       command=lambda: run_script("clear-ocr.py")).pack(pady=8)
Button(root, text="Datasheet Scraper", width=35, pady=8,
       command=lambda: run_script("datasheet-scraper.py")).pack(pady=8)

Label(root, text="(Each tool opens in a new window)", font=("Arial", 10), pady=20).pack(side=tk.BOTTOM)

root.mainloop()
