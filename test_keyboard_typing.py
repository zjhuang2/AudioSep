#!/usr/bin/env python3
"""
Quick test of keyboard typing sound separation
"""

import os
import sys
import torch
import torchaudio
import numpy as np
import soundfile as sf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pipeline import build_audiosep, separate_audio
from utils import calculate_sdr


def mix_audio_at_snr(target_audio, noise_audio, snr_db):
    """Mix target and noise at specified SNR."""
    min_len = min(target_audio.shape[-1], noise_audio.shape[-1])
    target_audio = target_audio[..., :min_len]
    noise_audio = noise_audio[..., :min_len]
    
    target_power = torch.mean(target_audio ** 2)
    noise_power = torch.mean(noise_audio ** 2)
    
    snr_linear = 10 ** (snr_db / 10)
    noise_scale = torch.sqrt(target_power / (noise_power * snr_linear))
    
    mixed = target_audio + noise_scale * noise_audio
    
    max_val = torch.max(torch.abs(mixed))
    if max_val > 1.0:
        mixed = mixed / max_val * 0.95
        
    return mixed


def main():
    print("Testing Keyboard Typing Separation")
    print("=" * 35)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_audiosep(
        config_yaml='config/audiosep_base.yaml',
        checkpoint_path='checkpoint/audiosep_base_4M_steps.ckpt',
        device=device
    )
    
    # Load files
    target_file = "test_samples/triggering_sounds/keyboard_typing/keyboard_typing_000.wav"
    noise_file = "test_samples/background_noises/crowd_noise/crowd_noise_000.wav"
    
    target_audio, sr = torchaudio.load(target_file)
    noise_audio, _ = torchaudio.load(noise_file)
    
    # Test at 0 dB SNR
    mixed_audio = mix_audio_at_snr(target_audio, noise_audio, 0)
    
    # Save and process
    sf.write("keyboard_mix.wav", mixed_audio.squeeze().cpu().numpy(), sr)
    
    # Test different queries
    queries = ["keyboard typing", "typing", "computer keyboard", "keys"]
    
    for query in queries:
        separated_path = f"keyboard_{query.replace(' ', '_')}.wav"
        
        try:
            separate_audio(
                model=model,
                audio_file="keyboard_mix.wav", 
                text=query,
                output_file=separated_path,
                device=device
            )
            
            separated_audio, _ = torchaudio.load(separated_path)
            
            # Calculate improvement
            min_len = min(target_audio.shape[-1], separated_audio.shape[-1], mixed_audio.shape[-1])
            target = target_audio[..., :min_len].numpy()
            separated = separated_audio[..., :min_len].numpy()
            mixture = mixed_audio[..., :min_len].numpy()
            
            original_sdr = calculate_sdr(ref=target, est=mixture)
            separated_sdr = calculate_sdr(ref=target, est=separated)
            improvement = separated_sdr - original_sdr
            
            correlation = np.corrcoef(target.flatten(), separated.flatten())[0, 1]
            
            print(f"Query: '{query}'")
            print(f"  SDR Improvement: {improvement:+5.1f} dB")
            print(f"  Correlation: {correlation:.3f}")
            
            # Clean up
            os.remove(separated_path)
            
        except Exception as e:
            print(f"Query: '{query}' - Error: {e}")
    
    # Clean up
    os.remove("keyboard_mix.wav")


if __name__ == "__main__":
    main()