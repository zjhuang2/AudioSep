#!/usr/bin/env python3
"""
Command-line interactive real-time audio processor
"""
import torch
from pipeline import build_audiosep
from realtime_extract_sound import RealtimeSoundExtractor
from realtime_audio_filter import RealtimeAudioFilter

def main():
    print("=" * 60)
    print("AudioSep Real-time Audio Processor")
    print("=" * 60)
    
    # Get user input
    print("\nWhat sound do you want to process?")
    print("Examples: human voice, music, dog barking, keyboard typing, background noise")
    sound_description = input("Enter sound description: ").strip()
    
    if not sound_description:
        print("No sound description provided. Exiting.")
        return
    
    print("\nSelect mode:")
    print("1. Remove - subtract this sound from audio (everything else passes through)")
    print("2. Extract - keep only this sound (everything else is filtered out)")
    
    while True:
        mode_choice = input("Enter 1 or 2: ").strip()
        if mode_choice in ['1', '2']:
            break
        print("Invalid choice. Please enter 1 or 2.")
    
    mode = 'remove' if mode_choice == '1' else 'extract'
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load model
    print("Loading AudioSep model...")
    try:
        model = build_audiosep(
            config_yaml='config/audiosep_base.yaml',
            checkpoint_path='checkpoint/audiosep_base_4M_steps.ckpt',
            device=device
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create processor
    print("\n" + "=" * 60)
    if mode == 'remove':
        print(f"REMOVE MODE: Will remove '{sound_description}' from microphone input")
        processor = RealtimeAudioFilter(
            model=model,
            text_query=sound_description,
            device=device,
            chunk_duration=2.0,
            overlap=0.5
        )
    else:
        print(f"EXTRACT MODE: Will extract only '{sound_description}' from microphone input")
        processor = RealtimeSoundExtractor(
            model=model,
            text_query=sound_description,
            device=device,
            chunk_duration=2.0,
            overlap=0.5
        )
    
    print("=" * 60)
    print("\nStarting real-time processing...")
    print("Speak into your microphone to test")
    print("Press Ctrl+C to stop\n")
    
    # Start processing
    try:
        processor.start()
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    main()