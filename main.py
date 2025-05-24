import tkinter as tk
from tkinter import Button, Label, Entry, StringVar, OptionMenu, Toplevel
import subprocess
import sys
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import cv2
from PIL import Image, ImageTk
import webbrowser

# Helper to run a script in a new process
def run_script(script_name):
    python_exe = sys.executable
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    subprocess.Popen([python_exe, script_path])

def get_video_devices():
    # Try /dev/video[0-9] up to 10 devices
    devices = []
    for i in range(10):
        if os.path.exists(f"/dev/video{i}"):
            devices.append(f"/dev/video{i}")
    return devices

def show_video_feed(device):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        tk.messagebox.showerror("Error", f"Cannot open video device: {device}")
        return
    win = Toplevel(root)
    win.title(f"Video Feed: {device}")
    l = Label(win)
    l.pack()
    def update():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img.thumbnail((640, 480))
            imgtk = ImageTk.PhotoImage(img)
            l.imgtk = imgtk
            l.config(image=imgtk)
        if win.winfo_exists():
            win.after(30, update)
        else:
            cap.release()
    update()

root = tk.Tk()
root.title("Microscope-OCR Main Launcher")
root.geometry("400x400")

Label(root, text="Microscope-OCR Toolkit", font=("Arial", 16, "bold"), pady=20).pack()

# Datasheet search input
search_frame = tk.Frame(root)
search_frame.pack(pady=5)
search_var = StringVar()
Entry(search_frame, textvariable=search_var, width=22, font=("Arial", 12)).pack(side=tk.LEFT, padx=5)
Button(search_frame, text="Search Datasheet", width=16, command=lambda: run_script_with_arg("datasheet-scraper.py", search_var.get())).pack(side=tk.LEFT)

def run_script_with_arg(script_name, arg):
    python_exe = sys.executable
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    if arg:
        # Capture output and show in GUI
        proc = subprocess.Popen([python_exe, script_path, arg], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        def check_output():
            out, err = proc.communicate()
            result = out.strip() if out else err.strip()
            # Try to extract a URL from the output
            url = None
            for line in result.splitlines():
                if line.startswith("Selected Link:"):
                    url = line.split("Selected Link:", 1)[-1].strip()
                    break
            show_datasheet_result(result, url)
        root.after(100, check_output)
    else:
        subprocess.Popen([python_exe, script_path])

def show_datasheet_result(result, url=None):
    win = Toplevel(root)
    win.title("Datasheet Search Result")
    label = Label(win, text=result, wraplength=500, justify=tk.LEFT, padx=10, pady=10, fg="blue" if url else None, cursor="hand2" if url else None)
    label.pack()
    if url:
        import threading
        def open_url(event=None):
            threading.Thread(target=lambda: webbrowser.open(url, new=2)).start()
        # Bind Ctrl+Click to open the link in the default browser
        label.bind('<Control-Button-1>', open_url)
        Label(win, text="Ctrl+Click the link above to open in your default browser", font=("Arial", 9), fg="gray").pack(pady=(5,0))

# Video input selection
video_devices = ["None"] + get_video_devices()
video_var = StringVar(value=video_devices[0])
video_frame = tk.Frame(root)
video_frame.pack(pady=5)
Label(video_frame, text="Video Input:").pack(side=tk.LEFT, padx=5)
OptionMenu(video_frame, video_var, *video_devices).pack(side=tk.LEFT)
Button(video_frame, text="Show Feed", command=lambda: show_video_feed(video_var.get()) if video_var.get() != "None" else None).pack(side=tk.LEFT, padx=5)

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
