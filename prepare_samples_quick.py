#!/usr/bin/env python3
"""
Quick sample preparation - downloads just a few samples for testing
"""

import os
import json
import subprocess
import pandas as pd
from pathlib import Path
from typing import Dict, List

# Just a few sounds for quick testing
TEST_SOUNDS = {
    "chewing": ["Chewing, mastication"],
    "keyboard_typing": ["Computer keyboard"],
    "dog_barking": ["Bark", "Dog"],
    "snoring": ["Snoring"],
    "water_dripping": ["Drip"]
}

TEST_NOISES = {
    "traffic_noise": ["Traffic noise, roadway noise"],
    "crowd_noise": ["Crowd"]
}


def download_youtube_segment(ytid: str, start: float, end: float, output_path: str) -> bool:
    """Download a segment from YouTube."""
    try:
        url = f"https://www.youtube.com/watch?v={ytid}"
        temp_file = "temp_audio.webm"
        
        print(f"  Downloading {ytid} ({start}-{end}s)...")
        
        # Download with yt-dlp (quiet mode)
        result = subprocess.run(
            ["yt-dlp", "-f", "bestaudio", url, "-o", temp_file, "--quiet"],
            capture_output=True
        )
        
        if result.returncode != 0:
            print(f"  Failed to download {ytid}")
            return False
        
        # Extract segment with ffmpeg
        duration = end - start
        result = subprocess.run([
            "ffmpeg", "-i", temp_file, 
            "-ss", str(start), "-t", str(duration),
            "-ar", "32000",  # 32kHz sample rate
            "-ac", "1",      # Mono
            "-loglevel", "error",
            output_path
        ])
        
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        if os.path.exists(output_path):
            print(f"  ✓ Saved to {output_path}")
            return True
        else:
            print(f"  Failed to extract segment")
            return False
            
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    # Load AudioSet metadata
    print("Loading AudioSet metadata...")
    eval_df = pd.read_csv(
        "eval_segments.csv",
        skiprows=3,
        header=None,
        names=['YTID', 'start_seconds', 'end_seconds', 'positive_labels'],
        skipinitialspace=True,
        quotechar='"'
    )
    
    # Load class labels
    labels_df = pd.read_csv("class_labels_indices.csv")
    label_to_mid = dict(zip(labels_df['display_name'], labels_df['mid']))
    
    # Create output directory
    output_dir = Path("test_samples")
    output_dir.mkdir(exist_ok=True)
    
    trigger_samples = {}
    noise_samples = {}
    
    # Download triggering sounds (just 2 samples each)
    print("\nDownloading triggering sound samples...")
    for sound_name, class_names in TEST_SOUNDS.items():
        print(f"\n{sound_name}:")
        sound_dir = output_dir / "triggering_sounds" / sound_name
        sound_dir.mkdir(parents=True, exist_ok=True)
        
        samples = []
        for class_name in class_names:
            if class_name in label_to_mid:
                mid = label_to_mid[class_name]
                matching = eval_df[eval_df['positive_labels'].str.contains(mid, na=False)]
                
                if len(matching) > 0:
                    # Take just 2 samples
                    for idx, (_, row) in enumerate(matching.head(2).iterrows()):
                        output_path = sound_dir / f"{sound_name}_{idx:03d}.wav"
                        if download_youtube_segment(
                            row['YTID'], 
                            row['start_seconds'], 
                            row['end_seconds'],
                            str(output_path)
                        ):
                            samples.append(str(output_path))
                    break
        
        trigger_samples[sound_name] = samples
    
    # Download background noises (just 2 samples each)
    print("\n\nDownloading background noise samples...")
    for noise_name, class_names in TEST_NOISES.items():
        print(f"\n{noise_name}:")
        noise_dir = output_dir / "background_noises" / noise_name.replace(' ', '_')
        noise_dir.mkdir(parents=True, exist_ok=True)
        
        samples = []
        for class_name in class_names:
            if class_name in label_to_mid:
                mid = label_to_mid[class_name]
                matching = eval_df[eval_df['positive_labels'].str.contains(mid, na=False)]
                
                if len(matching) > 0:
                    for idx, (_, row) in enumerate(matching.head(2).iterrows()):
                        output_path = noise_dir / f"{noise_name.replace(' ', '_')}_{idx:03d}.wav"
                        if download_youtube_segment(
                            row['YTID'],
                            row['start_seconds'],
                            row['end_seconds'],
                            str(output_path)
                        ):
                            samples.append(str(output_path))
                    break
        
        noise_samples[noise_name] = samples
    
    # Save mappings
    with open(output_dir / "trigger_samples.json", 'w') as f:
        json.dump(trigger_samples, f, indent=2)
    
    with open(output_dir / "noise_samples.json", 'w') as f:
        json.dump(noise_samples, f, indent=2)
    
    print(f"\n\nDownload complete!")
    print(f"Trigger samples: {output_dir}/trigger_samples.json")
    print(f"Noise samples: {output_dir}/noise_samples.json")
    
    # Summary
    total_trigger = sum(len(v) for v in trigger_samples.values())
    total_noise = sum(len(v) for v in noise_samples.values())
    print(f"\nDownloaded {total_trigger} triggering sounds and {total_noise} background noises")
    
    print(f"\nYou can now run a quick test with:")
    print(f"python noise_sensitivity_evaluation.py \\")
    print(f"    --audio-samples-json {output_dir}/trigger_samples.json \\")
    print(f"    --noise-samples-json {output_dir}/noise_samples.json \\")
    print(f"    --output-dir quick_test_results")


if __name__ == "__main__":
    main()