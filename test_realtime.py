#!/usr/bin/env python3
"""
Simple test script to check real-time audio functionality
"""
import sounddevice as sd
import numpy as np

print("Available audio devices:")
print(sd.query_devices())
print("\nDefault input device:", sd.default.device[0])
print("Default output device:", sd.default.device[1])

# Test recording
print("\nTesting microphone access...")
try:
    duration = 1  # seconds
    fs = 32000
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    print(f"Successfully recorded {len(recording)} samples")
    print(f"Audio range: [{np.min(recording):.3f}, {np.max(recording):.3f}]")
except Exception as e:
    print(f"Error accessing microphone: {e}")

print("\nMicrophone test complete!")