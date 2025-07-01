#!/usr/bin/env python3
"""
Quick test to demonstrate AudioSep on noisy contexts
Tests just one sound (chewing) in different noise conditions
"""

import os
import sys
import json
import torch
import torchaudio
import numpy as np
import soundfile as sf
from pathlib import Path

# Add AudioSep to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pipeline import build_audiosep, separate_audio, remove_audio
from utils import calculate_sdr, calculate_sisdr


def mix_audio_at_snr(target_audio, noise_audio, snr_db):
    """Mix target and noise at specified SNR."""
    # Ensure same length
    min_len = min(target_audio.shape[-1], noise_audio.shape[-1])
    target_audio = target_audio[..., :min_len]
    noise_audio = noise_audio[..., :min_len]
    
    # Calculate power
    target_power = torch.mean(target_audio ** 2)
    noise_power = torch.mean(noise_audio ** 2)
    
    # Calculate scaling factor for desired SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_scale = torch.sqrt(target_power / (noise_power * snr_linear))
    
    # Mix signals
    mixed = target_audio + noise_scale * noise_audio
    
    # Normalize to prevent clipping
    max_val = torch.max(torch.abs(mixed))
    if max_val > 1.0:
        mixed = mixed / max_val * 0.95
        
    return mixed


def evaluate_separation(original, separated, mixture):
    """Calculate metrics."""
    # Ensure same length
    min_len = min(original.shape[-1], separated.shape[-1], mixture.shape[-1])
    original = original[..., :min_len].numpy()
    separated = separated[..., :min_len].numpy()
    mixture = mixture[..., :min_len].numpy()
    
    # Calculate SDR improvement
    try:
        original_sdr = calculate_sdr(ref=original, est=mixture)
        separated_sdr = calculate_sdr(ref=original, est=separated)
        sdr_improvement = separated_sdr - original_sdr
    except:
        sdr_improvement = 0.0
    
    # Calculate correlation
    correlation = np.corrcoef(original.flatten(), separated.flatten())[0, 1]
    
    return {
        'original_sdr': float(original_sdr),
        'separated_sdr': float(separated_sdr),
        'sdr_improvement': float(sdr_improvement),
        'correlation': float(correlation)
    }


def main():
    print("Quick AudioSep Noisy Context Test")
    print("=" * 40)
    
    # Load AudioSep model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = build_audiosep(
        config_yaml='config/audiosep_base.yaml',
        checkpoint_path='checkpoint/audiosep_base_4M_steps.ckpt',
        device=device
    )
    
    # Load audio files
    target_file = "test_samples/triggering_sounds/chewing/chewing_000.wav"
    noise_file = "test_samples/background_noises/traffic_noise/traffic_noise_000.wav"
    
    if not os.path.exists(target_file) or not os.path.exists(noise_file):
        print("Error: Audio files not found. Run prepare_samples_quick.py first.")
        return
    
    print(f"Target sound: {target_file}")
    print(f"Noise sound: {noise_file}")
    
    # Load audio
    target_audio, sr = torchaudio.load(target_file)
    noise_audio, _ = torchaudio.load(noise_file)
    
    # Test different SNR levels
    snr_levels = [-5, 0, 5, 10]
    results = []
    
    for snr_db in snr_levels:
        print(f"\nTesting SNR: {snr_db} dB")
        
        # Mix audio
        mixed_audio = mix_audio_at_snr(target_audio, noise_audio, snr_db)
        
        # Save mixed audio
        mixed_path = f"test_mix_snr{snr_db}.wav"
        sf.write(mixed_path, mixed_audio.squeeze().cpu().numpy(), sr)
        
        # Test separation
        separated_path = f"test_separated_snr{snr_db}.wav"
        eliminate_path = f"test_eliminated_snr{snr_db}.wav"
        
        try:
            # Separate the chewing sound
            separate_audio(
                model=model,
                audio_file=mixed_path,
                text="chewing",
                output_file=separated_path,
                device=device
            )
            
            # Remove the chewing sound
            remove_audio(
                model=model,
                audio_file=mixed_path,
                text="chewing",
                output_file=eliminate_path,
                device=device
            )
            
            # Load results
            separated_audio, _ = torchaudio.load(separated_path)
            
            # Evaluate
            metrics = evaluate_separation(target_audio, separated_audio, mixed_audio)
            metrics['snr_db'] = snr_db
            results.append(metrics)
            
            print(f"  Original SNR: {metrics['original_sdr']:.1f} dB")
            print(f"  Separated SNR: {metrics['separated_sdr']:.1f} dB")
            print(f"  SDR Improvement: {metrics['sdr_improvement']:.1f} dB")
            print(f"  Correlation: {metrics['correlation']:.3f}")
            
        except Exception as e:
            print(f"  Error: {e}")
            
        # Clean up
        for file in [mixed_path, separated_path, eliminate_path]:
            if os.path.exists(file):
                os.remove(file)
    
    # Summary
    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    
    successful_results = [r for r in results if 'sdr_improvement' in r]
    if successful_results:
        avg_improvement = np.mean([r['sdr_improvement'] for r in successful_results])
        avg_correlation = np.mean([r['correlation'] for r in successful_results])
        
        print(f"Average SDR Improvement: {avg_improvement:.1f} dB")
        print(f"Average Correlation: {avg_correlation:.3f}")
        
        print("\nResults by SNR:")
        for result in successful_results:
            print(f"  {result['snr_db']:2d} dB SNR: {result['sdr_improvement']:+5.1f} dB improvement")
    
    print("\nFiles tested:")
    print(f"  Target: {target_file}")
    print(f"  Noise: {noise_file}")


if __name__ == "__main__":
    main()