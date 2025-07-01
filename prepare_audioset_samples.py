#!/usr/bin/env python3
"""
Helper script to prepare AudioSet samples for noise sensitivity evaluation
Downloads and organizes audio samples for triggering sounds and background noises
"""

import os
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import requests
from tqdm import tqdm


# AudioSet class mappings for our triggering sounds
AUDIOSET_MAPPINGS = {
    # Eating/Mouth sounds
    "chewing": ["Chewing, mastication"],
    "slurping": ["Slurp", "Sip"],
    "lip_smacking": ["Smack, smacking lips"],
    "swallowing": ["Gulp", "Swallow"],
    
    # Breathing/Nasal sounds
    "sniffing": ["Sniff", "Sniffle"],
    "heavy_breathing": ["Breathing", "Pant"],
    "snoring": ["Snoring"],
    "throat_clearing": ["Throat clearing"],
    
    # Repetitive sounds
    "pen_clicking": ["Clicking", "Click"],
    "keyboard_typing": ["Typing", "Computer keyboard"],
    "finger_tapping": ["Tapping", "Finger snapping"],
    "foot_tapping": ["Footsteps", "Tap"],
    
    # Environmental sounds
    "clock_ticking": ["Tick-tock", "Tick"],
    "water_dripping": ["Drip", "Water tap, faucet"],
    "dog_barking": ["Bark", "Dog"],
    "baby_crying": ["Baby cry, infant cry", "Crying, sobbing"],
    
    # Electronic sounds
    "phone_notification": ["Telephone bell ringing", "Ringtone"],
    "buzzing": ["Buzz", "Hum"],
    
    # Mechanical sounds
    "vacuum_cleaner": ["Vacuum cleaner"],
    "lawn_mower": ["Lawn mower"]
}

# Background noise mappings
NOISE_MAPPINGS = {
    "background conversation": ["Conversation", "Speech", "Chatter"],
    "traffic noise": ["Traffic noise, roadway noise", "Vehicle"],
    "office ambience": ["Inside, small room", "Office"],
    "cafe ambience": ["Restaurant", "Crowd"],
    "street noise": ["Outside, urban or manmade", "Street music"],
    "crowd noise": ["Crowd", "Hubbub, speech noise, speech babble"]
}


class AudioSetSamplePreparer:
    def __init__(self, audioset_csv_path: str, output_dir: str):
        """Initialize with AudioSet metadata CSV."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load AudioSet metadata
        print("Loading AudioSet metadata...")
        # Read the CSV with proper handling of quoted fields
        self.audioset_df = pd.read_csv(
            audioset_csv_path,
            skiprows=3,  # Skip the comment lines
            header=None,  # No header in data
            names=['YTID', 'start_seconds', 'end_seconds', 'positive_labels'],
            skipinitialspace=True,  # Handle spaces after commas
            quotechar='"'  # Handle quoted fields with commas
        )
        
        # Load class labels
        self.class_labels = self._load_class_labels()
        
    def _load_class_labels(self) -> Dict[str, str]:
        """Load AudioSet class labels mapping."""
        # This would need the class_labels_indices.csv file
        # For now, return a simplified mapping
        labels_file = Path("evaluation/metadata/class_labels_indices.csv")
        if labels_file.exists():
            labels_df = pd.read_csv(labels_file)
            return dict(zip(labels_df['display_name'], labels_df['mid']))
        else:
            print("Warning: class_labels_indices.csv not found")
            return {}
    
    def find_samples_for_sound(self, sound_classes: List[str], num_samples: int = 5) -> List[Dict]:
        """Find AudioSet samples containing specified sound classes."""
        samples = []
        
        for class_name in sound_classes:
            if class_name in self.class_labels:
                class_id = self.class_labels[class_name]
                
                # Find rows containing this class
                matching_rows = self.audioset_df[
                    self.audioset_df['positive_labels'].str.contains(class_id, na=False)
                ]
                
                # Take random samples
                if len(matching_rows) > 0:
                    sample_rows = matching_rows.sample(n=min(num_samples, len(matching_rows)))
                    
                    for _, row in sample_rows.iterrows():
                        samples.append({
                            'ytid': row['YTID'],
                            'start': row['start_seconds'],
                            'end': row['end_seconds'],
                            'class': class_name
                        })
        
        return samples
    
    def download_youtube_segment(self, ytid: str, start: float, end: float, 
                               output_path: str) -> bool:
        """Download a segment from YouTube."""
        import subprocess
        
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
            ], capture_output=True)
            
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:  # At least 1KB
                print(f"  ✓ Saved to {output_path}")
                return True
            else:
                print(f"  Failed to extract segment")
                return False
                
        except Exception as e:
            print(f"  Error: {e}")
            # Clean up in case of error
            if os.path.exists("temp_audio.webm"):
                os.remove("temp_audio.webm")
            return False
    
    def prepare_all_samples(self) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """Prepare all audio samples for evaluation."""
        trigger_samples = {}
        noise_samples = {}
        
        # Prepare triggering sound samples
        print("\nPreparing triggering sound samples...")
        for sound_name, class_names in AUDIOSET_MAPPINGS.items():
            print(f"Finding samples for {sound_name}...")
            samples = self.find_samples_for_sound(class_names, num_samples=5)
            
            if samples:
                sound_dir = self.output_dir / "triggering_sounds" / sound_name
                sound_dir.mkdir(parents=True, exist_ok=True)
                
                sample_paths = []
                for i, sample in enumerate(samples):
                    output_path = sound_dir / f"{sound_name}_{i:03d}.wav"
                    if self.download_youtube_segment(
                        sample['ytid'], sample['start'], sample['end'], 
                        str(output_path)
                    ):
                        sample_paths.append(str(output_path))
                
                trigger_samples[sound_name] = sample_paths
        
        # Prepare background noise samples
        print("\nPreparing background noise samples...")
        for noise_name, class_names in NOISE_MAPPINGS.items():
            print(f"Finding samples for {noise_name}...")
            samples = self.find_samples_for_sound(class_names, num_samples=3)
            
            if samples:
                noise_dir = self.output_dir / "background_noises" / noise_name.replace(' ', '_')
                noise_dir.mkdir(parents=True, exist_ok=True)
                
                sample_paths = []
                for i, sample in enumerate(samples):
                    output_path = noise_dir / f"{noise_name.replace(' ', '_')}_{i:03d}.wav"
                    if self.download_youtube_segment(
                        sample['ytid'], sample['start'], sample['end'],
                        str(output_path)
                    ):
                        sample_paths.append(str(output_path))
                
                noise_samples[noise_name] = sample_paths
        
        # Save sample mappings
        with open(self.output_dir / "trigger_samples.json", 'w') as f:
            json.dump(trigger_samples, f, indent=2)
        
        with open(self.output_dir / "noise_samples.json", 'w') as f:
            json.dump(noise_samples, f, indent=2)
        
        return trigger_samples, noise_samples
    
    def create_sample_dataset_from_local(self, audio_dir: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """Create sample mappings from local audio files (alternative to downloading)."""
        trigger_samples = {}
        noise_samples = {}
        
        audio_path = Path(audio_dir)
        
        # Map any existing audio files to our categories
        print("Creating sample dataset from local files...")
        
        # This is a simplified version - in practice, you'd want to:
        # 1. Use an audio classifier to identify which sounds are in each file
        # 2. Or manually organize files into folders by sound type
        # 3. Or use AudioSet's pre-computed embeddings
        
        # For now, create empty structure
        for sound_name in AUDIOSET_MAPPINGS.keys():
            trigger_samples[sound_name] = []
        
        for noise_name in NOISE_MAPPINGS.keys():
            noise_samples[noise_name] = []
        
        # Save mappings
        with open(self.output_dir / "trigger_samples.json", 'w') as f:
            json.dump(trigger_samples, f, indent=2)
        
        with open(self.output_dir / "noise_samples.json", 'w') as f:
            json.dump(noise_samples, f, indent=2)
        
        return trigger_samples, noise_samples


def main():
    parser = argparse.ArgumentParser(description="Prepare AudioSet samples for evaluation")
    parser.add_argument('--audioset-csv', type=str, 
                      default='audioset_train_strong.csv',
                      help='Path to AudioSet CSV file')
    parser.add_argument('--output-dir', type=str, default='audioset_samples',
                      help='Output directory for audio samples')
    parser.add_argument('--use-local', action='store_true',
                      help='Use local audio files instead of downloading')
    parser.add_argument('--local-audio-dir', type=str,
                      help='Directory containing local audio files')
    
    args = parser.parse_args()
    
    preparer = AudioSetSamplePreparer(args.audioset_csv, args.output_dir)
    
    if args.use_local and args.local_audio_dir:
        trigger_samples, noise_samples = preparer.create_sample_dataset_from_local(
            args.local_audio_dir
        )
    else:
        trigger_samples, noise_samples = preparer.prepare_all_samples()
    
    print(f"\nSample preparation complete!")
    print(f"Trigger samples saved to: {args.output_dir}/trigger_samples.json")
    print(f"Noise samples saved to: {args.output_dir}/noise_samples.json")
    print(f"\nYou can now run the evaluation with:")
    print(f"python noise_sensitivity_evaluation.py \\")
    print(f"    --audio-samples-json {args.output_dir}/trigger_samples.json \\")
    print(f"    --noise-samples-json {args.output_dir}/noise_samples.json")


if __name__ == "__main__":
    main()