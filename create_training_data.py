#!/usr/bin/env python3
"""
Script to create training data JSON for AudioSep finetuning
"""

import os
import json
import argparse
from pathlib import Path


def create_training_json(audio_dir, output_json, caption_type="filename"):
    """
    Create a training JSON file from a directory of audio files.
    
    Args:
        audio_dir: Directory containing audio files
        output_json: Output JSON file path
        caption_type: How to generate captions - "filename", "manual", or "csv"
    """
    
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
    audio_files = []
    
    # Collect all audio files
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if Path(file).suffix.lower() in audio_extensions:
                audio_files.append(os.path.join(root, file))
    
    print(f"Found {len(audio_files)} audio files")
    
    # Create data entries
    data_entries = []
    
    for audio_path in sorted(audio_files):
        filename = Path(audio_path).stem
        
        if caption_type == "filename":
            # Use filename as caption (replace underscores/hyphens with spaces)
            caption = filename.replace('_', ' ').replace('-', ' ')
        elif caption_type == "manual":
            # Ask for manual caption
            print(f"\nAudio file: {audio_path}")
            caption = input("Enter caption (or press Enter to use filename): ").strip()
            if not caption:
                caption = filename.replace('_', ' ').replace('-', ' ')
        else:
            # Default to filename
            caption = filename.replace('_', ' ').replace('-', ' ')
        
        data_entries.append({
            "wav": audio_path,
            "caption": caption
        })
    
    # Save to JSON
    output_data = {"data": data_entries}
    
    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nCreated training JSON with {len(data_entries)} entries")
    print(f"Saved to: {output_json}")
    
    # Show sample entries
    print("\nSample entries:")
    for entry in data_entries[:3]:
        print(f"  Audio: {entry['wav']}")
        print(f"  Caption: {entry['caption']}\n")


def create_from_csv(csv_file, audio_dir, output_json):
    """
    Create training JSON from a CSV file with columns: filename, caption
    """
    import csv
    
    data_entries = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_path = os.path.join(audio_dir, row['filename'])
            if os.path.exists(audio_path):
                data_entries.append({
                    "wav": audio_path,
                    "caption": row['caption']
                })
            else:
                print(f"Warning: {audio_path} not found")
    
    output_data = {"data": data_entries}
    
    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nCreated training JSON with {len(data_entries)} entries from CSV")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create AudioSep training data JSON")
    parser.add_argument("audio_dir", help="Directory containing audio files")
    parser.add_argument("output_json", help="Output JSON file path")
    parser.add_argument("--caption-type", choices=["filename", "manual", "csv"], 
                       default="filename", help="How to generate captions")
    parser.add_argument("--csv-file", help="CSV file with filename,caption columns (for csv mode)")
    
    args = parser.parse_args()
    
    if args.caption_type == "csv":
        if not args.csv_file:
            print("Error: --csv-file required when using csv caption type")
            exit(1)
        create_from_csv(args.csv_file, args.audio_dir, args.output_json)
    else:
        create_training_json(args.audio_dir, args.output_json, args.caption_type)