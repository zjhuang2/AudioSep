#!/usr/bin/env python3
"""
Script to fix the noise sample mapping between prepare_audioset_samples.py output
and what noise_sensitivity_evaluation.py expects
"""

import json
import sys
from pathlib import Path

def fix_noise_mapping(original_noise_json: str, output_noise_json: str):
    """Fix the noise sample mapping."""
    
    # Load the original noise samples
    with open(original_noise_json, 'r') as f:
        original_noise = json.load(f)
    
    # Create the expected mapping
    fixed_noise = {
        "background conversation": original_noise.get("background_conversation", []),
        "traffic noise": original_noise.get("traffic_noise", []),
        "office ambience": original_noise.get("office_ambience", []),
        "cafe ambience": original_noise.get("cafe_ambience", []),
        "street noise": original_noise.get("street_noise", []),
        "crowd noise": original_noise.get("crowd_noise", [])
    }
    
    # If we don't have enough variety, reuse what we have
    available_samples = []
    for samples in original_noise.values():
        available_samples.extend(samples)
    
    # Fill in missing mappings with available samples
    if not fixed_noise["background conversation"] and available_samples:
        fixed_noise["background conversation"] = [available_samples[0]] if len(available_samples) > 0 else []
    
    if not fixed_noise["traffic noise"] and available_samples:
        fixed_noise["traffic noise"] = [available_samples[1]] if len(available_samples) > 1 else available_samples[:1]
        
    if not fixed_noise["office ambience"] and available_samples:
        fixed_noise["office ambience"] = [available_samples[0]] if len(available_samples) > 0 else []
        
    if not fixed_noise["cafe ambience"] and available_samples:
        fixed_noise["cafe ambience"] = [available_samples[1]] if len(available_samples) > 1 else available_samples[:1]
        
    if not fixed_noise["street noise"] and available_samples:
        fixed_noise["street noise"] = available_samples[:1]
        
    if not fixed_noise["crowd noise"] and available_samples:
        fixed_noise["crowd noise"] = available_samples[:1]
    
    # Save the fixed mapping
    with open(output_noise_json, 'w') as f:
        json.dump(fixed_noise, f, indent=2)
    
    print(f"Fixed noise mapping saved to: {output_noise_json}")
    print("Mappings:")
    for noise_type, samples in fixed_noise.items():
        print(f"  {noise_type}: {len(samples)} samples")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fix_noise_mapping.py <input_noise.json> <output_noise.json>")
        sys.exit(1)
    
    fix_noise_mapping(sys.argv[1], sys.argv[2])