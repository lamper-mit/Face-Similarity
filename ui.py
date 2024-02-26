import tkinter as tk
from tkinter import filedialog, messagebox, Canvas, Scrollbar, Frame, ttk
from compare import verify_and_copy
import json
from PIL import Image, ImageTk
import os

def select_directory(entry):
    directory = filedialog.askdirectory()
    entry.delete(0, tk.END)
    entry.insert(0, directory)

def start_comparison():
    source = source_entry.get()
    target = target_entry.get()
    reference = reference_entry.get()
    try:
        cutoff = float(cutoff_entry.get())
    except ValueError:
        messagebox.showerror("Error", "Invalid cutoff value")
        return

    try:
        # Start the comparison process
        verify_and_copy(source, target, reference, cutoff)
        # Update the progress bar to 100% when the comparison process is finished
        progress_bar['value'] = 100
    except Exception as e:
        messagebox.showerror("Error", str(e))
        return

    # Load the similarity scores
    with open(os.path.join(target, 'similarity_scores.json'), 'r') as f:
        data = json.load(f)

    # Clear the canvas
    for widget in frame.winfo_children():
        widget.destroy()

    # Add each image and its score to the canvas
    for image_path, score in data['scores'].items():
        # Open the image and resize it to a smaller size for display
        img = Image.open(image_path)
        img = img.resize((100, 100), Image.Resampling.LANCZOS)

        # Convert the image to a format Tkinter can use
        tk_img = ImageTk.PhotoImage(img)

        # Add the image and score to the canvas
        label = tk.Label(frame, image=tk_img)
        label.image = tk_img  # Keep a reference to the image to prevent it from being garbage collected
        label.pack()
        tk.Label(frame, text=f"{os.path.basename(image_path)}: {score}").pack()

root = tk.Tk()

source_entry = tk.Entry(root)
source_entry.pack()
source_button = tk.Button(root, text="Select Source Directory", command=lambda: select_directory(source_entry))
source_button.pack()

target_entry = tk.Entry(root)
target_entry.pack()
target_button = tk.Button(root, text="Select Target Directory", command=lambda: select_directory(target_entry))
target_button.pack()

reference_entry = tk.Entry(root)
reference_entry.pack()
reference_button = tk.Button(root, text="Select Reference Directory", command=lambda: select_directory(reference_entry))
reference_button.pack()

# Create a progress bar
progress_bar = ttk.Progressbar(root, length=200, mode='determinate')
progress_bar.pack()

# Create a canvas to display the images and scores
canvas = Canvas(root)
scrollbar = Scrollbar(root, command=canvas.yview)
canvas.configure(yscrollcommand=scrollbar.set)
frame = Frame(canvas)
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")
canvas.create_window((0, 0), window=frame, anchor="nw")
frame.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))

cutoff_entry = tk.Entry(root)
cutoff_entry.pack()

start_button = tk.Button(root, text="Start Comparison", command=start_comparison)
start_button.pack()

root.mainloop()