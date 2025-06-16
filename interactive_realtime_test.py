#!/usr/bin/env python3
"""
Interactive real-time audio processor with simple UI
"""
import torch
import tkinter as tk
from tkinter import ttk
import threading
from pipeline import build_audiosep
from realtime_extract_sound import RealtimeSoundExtractor
from realtime_audio_filter import RealtimeAudioFilter

class AudioSepGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AudioSep Real-time Processor")
        self.root.geometry("500x400")
        
        # Model and processor
        self.model = None
        self.processor = None
        self.is_running = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create UI
        self.create_widgets()
        
        # Load model in background
        self.load_model_thread = threading.Thread(target=self.load_model)
        self.load_model_thread.start()
        
    def create_widgets(self):
        # Title
        title_label = tk.Label(self.root, text="AudioSep Real-time Audio Processor", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Status label
        self.status_label = tk.Label(self.root, text="Loading model...", 
                                    font=("Arial", 10), fg="blue")
        self.status_label.pack(pady=5)
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Sound description
        ttk.Label(main_frame, text="Sound to process:", font=("Arial", 12)).grid(row=0, column=0, sticky=tk.W, pady=10)
        self.sound_entry = ttk.Entry(main_frame, width=40, font=("Arial", 11))
        self.sound_entry.grid(row=0, column=1, pady=10, padx=10)
        self.sound_entry.insert(0, "background noise")
        
        # Examples
        examples_frame = ttk.Frame(main_frame)
        examples_frame.grid(row=1, column=0, columnspan=2, pady=5)
        ttk.Label(examples_frame, text="Examples: ", font=("Arial", 10, "italic")).pack(side=tk.LEFT)
        examples = ["human voice", "music", "dog barking", "keyboard typing", "traffic noise"]
        ttk.Label(examples_frame, text=", ".join(examples), font=("Arial", 10)).pack(side=tk.LEFT)
        
        # Mode selection
        ttk.Label(main_frame, text="Mode:", font=("Arial", 12)).grid(row=2, column=0, sticky=tk.W, pady=10)
        mode_frame = ttk.Frame(main_frame)
        mode_frame.grid(row=2, column=1, pady=10, sticky=tk.W)
        
        self.mode_var = tk.StringVar(value="remove")
        ttk.Radiobutton(mode_frame, text="Remove (subtract from audio)", 
                       variable=self.mode_var, value="remove",
                       command=self.update_description).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Extract (keep only this sound)", 
                       variable=self.mode_var, value="extract",
                       command=self.update_description).pack(anchor=tk.W)
        
        # Description
        self.desc_label = tk.Label(main_frame, text="", font=("Arial", 10), 
                                  fg="gray", wraplength=400)
        self.desc_label.grid(row=3, column=0, columnspan=2, pady=10)
        self.update_description()
        
        # Control buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=20)
        
        self.start_button = ttk.Button(button_frame, text="Start Processing", 
                                      command=self.toggle_processing, state="disabled")
        self.start_button.pack(side=tk.LEFT, padx=10)
        
        # Processing status
        self.processing_label = tk.Label(self.root, text="", font=("Arial", 11), fg="green")
        self.processing_label.pack(pady=10)
        
        # Instructions
        instructions = ("Instructions:\n"
                       "1. Enter a description of the sound you want to process\n"
                       "2. Choose whether to remove or extract that sound\n"
                       "3. Click 'Start Processing' to begin\n"
                       "4. Speak into your microphone and hear the processed output")
        tk.Label(self.root, text=instructions, font=("Arial", 9), 
                justify=tk.LEFT, fg="gray").pack(pady=10)
        
    def update_description(self):
        mode = self.mode_var.get()
        if mode == "remove":
            self.desc_label.config(text="The specified sound will be removed from your microphone input. "
                                      "Everything else will pass through.")
        else:
            self.desc_label.config(text="Only the specified sound will be extracted from your microphone input. "
                                      "Everything else will be filtered out.")
    
    def load_model(self):
        try:
            self.model = build_audiosep(
                config_yaml='config/audiosep_base.yaml',
                checkpoint_path='checkpoint/audiosep_base_4M_steps.ckpt',
                device=self.device
            )
            self.root.after(0, self.model_loaded)
        except Exception as e:
            self.root.after(0, lambda: self.model_load_error(str(e)))
    
    def model_loaded(self):
        self.status_label.config(text=f"Model loaded! Using device: {self.device}", fg="green")
        self.start_button.config(state="normal")
    
    def model_load_error(self, error):
        self.status_label.config(text=f"Error loading model: {error}", fg="red")
    
    def toggle_processing(self):
        if not self.is_running:
            self.start_processing()
        else:
            self.stop_processing()
    
    def start_processing(self):
        sound_description = self.sound_entry.get().strip()
        if not sound_description:
            self.processing_label.config(text="Please enter a sound description!", fg="red")
            return
        
        mode = self.mode_var.get()
        
        # Disable controls
        self.sound_entry.config(state="disabled")
        for child in self.root.winfo_children():
            if isinstance(child, ttk.Radiobutton):
                child.config(state="disabled")
        
        # Create processor
        if mode == "extract":
            self.processor = RealtimeSoundExtractor(
                model=self.model,
                text_query=sound_description,
                device=self.device,
                chunk_duration=0.5,
                overlap=0.1
            )
            action = "Extracting"
        else:
            self.processor = RealtimeAudioFilter(
                model=self.model,
                text_query=sound_description,
                device=self.device,
                chunk_duration=0.5,
                overlap=0.1
            )
            action = "Removing"
        
        # Update UI
        self.is_running = True
        self.start_button.config(text="Stop Processing")
        self.processing_label.config(text=f"{action} '{sound_description}' - Processing...", fg="green")
        
        # Start processing in separate thread
        self.processing_thread = threading.Thread(target=self.run_processor)
        self.processing_thread.start()
    
    def run_processor(self):
        try:
            self.processor.start()
        except Exception as e:
            self.root.after(0, lambda: self.processing_error(str(e)))
    
    def processing_error(self, error):
        self.processing_label.config(text=f"Processing error: {error}", fg="red")
        self.stop_processing()
    
    def stop_processing(self):
        if self.processor:
            self.processor.stop()
        
        # Re-enable controls
        self.sound_entry.config(state="normal")
        for child in self.root.winfo_children():
            if isinstance(child, ttk.Radiobutton):
                child.config(state="normal")
        
        # Update UI
        self.is_running = False
        self.start_button.config(text="Start Processing")
        self.processing_label.config(text="Processing stopped", fg="gray")
    
    def on_closing(self):
        if self.is_running:
            self.stop_processing()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = AudioSepGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()