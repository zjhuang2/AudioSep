#!/usr/bin/env python3
"""
AudioSep Feasibility Study for Noise-Sensitive Individuals
Evaluates AudioSep's performance in eliminating common triggering sounds
"""

import os
import sys
import json
import yaml
import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import argparse
import soundfile as sf
from sklearn.metrics import mean_squared_error

# Add AudioSep to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline import build_audiosep, separate_audio, remove_audio
from utils import calculate_sdr, calculate_sisdr


# Define 20 common triggering sounds for noise-sensitive individuals
# Based on misophonia and hyperacusis research
TRIGGERING_SOUNDS = {
    # Eating/Mouth sounds
    "chewing": ["chewing", "eating", "munching"],
    "slurping": ["slurping", "sipping", "drinking"],
    "lip_smacking": ["lip smacking", "smacking lips", "mouth sounds"],
    "swallowing": ["swallowing", "gulping"],
    
    # Breathing/Nasal sounds
    "sniffing": ["sniffing", "sniffling", "nose sounds"],
    "heavy_breathing": ["heavy breathing", "loud breathing", "panting"],
    "snoring": ["snoring", "sleep breathing"],
    "throat_clearing": ["throat clearing", "clearing throat", "ahem"],
    
    # Repetitive sounds
    "pen_clicking": ["pen clicking", "clicking pen", "click sound"],
    "keyboard_typing": ["keyboard typing", "typing", "keyboard sounds"],
    "finger_tapping": ["finger tapping", "tapping fingers", "drumming"],
    "foot_tapping": ["foot tapping", "tapping foot", "footsteps"],
    
    # Environmental sounds
    "clock_ticking": ["clock ticking", "ticking clock", "tick tock"],
    "water_dripping": ["water dripping", "dripping water", "drip sound"],
    "dog_barking": ["dog barking", "barking dog", "bark"],
    "baby_crying": ["baby crying", "crying baby", "infant crying"],
    
    # Electronic sounds
    "phone_notification": ["phone notification", "text message sound", "notification"],
    "buzzing": ["buzzing", "electrical buzz", "humming"],
    
    # Mechanical sounds
    "vacuum_cleaner": ["vacuum cleaner", "vacuuming", "vacuum sound"],
    "lawn_mower": ["lawn mower", "mowing", "lawn mowing"]
}

# Audio contexts for testing
AUDIO_CONTEXTS = {
    "quiet": {
        "description": "Target sound in quiet environment",
        "noise_sounds": [],
        "snr_db": None
    },
    "single_noise": {
        "description": "Target sound with one background noise",
        "noise_sounds": ["background conversation", "traffic noise", "office ambience"],
        "snr_db": [0, 5, 10]  # Different signal-to-noise ratios
    },
    "multiple_noises": {
        "description": "Target sound with multiple background noises",
        "noise_sounds": ["cafe ambience", "street noise", "crowd noise"],
        "snr_db": [-5, 0, 5]
    }
}


class NoiseSensitivityEvaluator:
    def __init__(self, model_config: str, checkpoint_path: str, output_dir: str):
        """Initialize the evaluator with AudioSep model."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load AudioSep model
        self.model = build_audiosep(
            config_yaml=model_config,
            checkpoint_path=checkpoint_path,
            device=self.device
        )
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.audio_samples_dir = self.output_dir / "audio_samples"
        self.mixed_audio_dir = self.output_dir / "mixed_audio"
        self.separated_audio_dir = self.output_dir / "separated_audio"
        self.results_dir = self.output_dir / "results"
        
        for dir in [self.audio_samples_dir, self.mixed_audio_dir, 
                   self.separated_audio_dir, self.results_dir]:
            dir.mkdir(exist_ok=True)
    
    def load_audio(self, audio_path: str, target_sr: int = 32000) -> Tuple[torch.Tensor, int]:
        """Load and resample audio to target sample rate."""
        waveform, sr = torchaudio.load(audio_path)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
        return waveform, target_sr
    
    def mix_audio_signals(self, target_audio: torch.Tensor, 
                         noise_audio: torch.Tensor, 
                         snr_db: float) -> torch.Tensor:
        """Mix target and noise audio at specified SNR."""
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
    
    def evaluate_separation(self, original: torch.Tensor, 
                          separated: torch.Tensor, 
                          mixture: torch.Tensor,
                          is_quiet_context: bool = False) -> Dict[str, float]:
        """Calculate evaluation metrics for separation quality."""
        # Ensure same length for comparison
        min_len = min(original.shape[-1], separated.shape[-1], mixture.shape[-1])
        original = original[..., :min_len].numpy()
        separated = separated[..., :min_len].numpy()
        mixture = mixture[..., :min_len].numpy()
        
        # Calculate metrics
        metrics = {}
        
        # SDR (Signal-to-Distortion Ratio)
        try:
            sdr_separated = calculate_sdr(ref=original, est=separated)
            metrics['sdr'] = float(sdr_separated)
            
            if not is_quiet_context:
                # Only calculate improvement for noisy contexts
                sdr_mixture = calculate_sdr(ref=original, est=mixture)
                metrics['sdr_improvement'] = float(sdr_separated - sdr_mixture)
                metrics['mixture_sdr'] = float(sdr_mixture)
            else:
                # For quiet context, show how much processing degrades the signal
                metrics['sdr_improvement'] = None
                metrics['mixture_sdr'] = None
                
        except Exception as e:
            metrics['sdr'] = 0.0
            metrics['sdr_improvement'] = 0.0
            metrics['mixture_sdr'] = 0.0
        
        # SI-SDR (Scale-Invariant SDR)
        try:
            sisdr_separated = calculate_sisdr(ref=original, est=separated)
            metrics['sisdr'] = float(sisdr_separated)
            
            if not is_quiet_context:
                sisdr_mixture = calculate_sisdr(ref=original, est=mixture)
                metrics['sisdr_improvement'] = float(sisdr_separated - sisdr_mixture)
                metrics['mixture_sisdr'] = float(sisdr_mixture)
            else:
                metrics['sisdr_improvement'] = None
                metrics['mixture_sisdr'] = None
                
        except Exception as e:
            metrics['sisdr'] = 0.0
            metrics['sisdr_improvement'] = 0.0
            metrics['mixture_sisdr'] = 0.0
        
        # MSE (Mean Squared Error) - always meaningful
        mse = mean_squared_error(original.flatten(), separated.flatten())
        metrics['mse'] = float(mse)
        
        # Correlation - always meaningful
        correlation = np.corrcoef(original.flatten(), separated.flatten())[0, 1]
        metrics['correlation'] = float(correlation) if not np.isnan(correlation) else 0.0
        
        return metrics
    
    def evaluate_elimination(self, original_mix: torch.Tensor,
                           eliminated: torch.Tensor,
                           target_sound: torch.Tensor) -> Dict[str, float]:
        """Evaluate how well the target sound was eliminated."""
        # Ensure same length
        min_len = min(original_mix.shape[-1], eliminated.shape[-1], target_sound.shape[-1])
        original_mix = original_mix[..., :min_len].numpy()
        eliminated = eliminated[..., :min_len].numpy()
        target_sound = target_sound[..., :min_len].numpy()
        
        metrics = {}
        
        # Calculate residual target sound energy
        # Ideally, correlation between eliminated and target should be low
        target_correlation = np.corrcoef(eliminated.flatten(), target_sound.flatten())[0, 1]
        metrics['target_residual_correlation'] = float(abs(target_correlation))
        
        # Energy reduction ratio
        original_energy = np.mean(original_mix ** 2)
        eliminated_energy = np.mean(eliminated ** 2)
        energy_reduction = 1 - (eliminated_energy / original_energy)
        metrics['energy_reduction_ratio'] = float(energy_reduction)
        
        # Spectral similarity between eliminated and original mix
        # Lower is better (more different from original)
        spectral_correlation = np.corrcoef(
            np.abs(np.fft.fft(eliminated.flatten())),
            np.abs(np.fft.fft(original_mix.flatten()))
        )[0, 1]
        metrics['spectral_difference'] = float(1 - spectral_correlation)
        
        return metrics
    
    def process_sound_in_context(self, sound_name: str, 
                               sound_queries: List[str],
                               target_audio_path: str,
                               context_name: str,
                               context_config: Dict,
                               noise_audio_paths: Optional[List[str]] = None) -> Dict:
        """Process a triggering sound in a specific audio context."""
        results = {
            'sound': sound_name,
            'context': context_name,
            'queries_tested': sound_queries,
            'evaluations': []
        }
        
        # Load target sound
        target_audio, sr = self.load_audio(target_audio_path)
        
        if context_name == "quiet":
            # Test in quiet environment (no mixing needed)
            for query in sound_queries:
                eval_result = self._evaluate_single_case(
                    sound_name=sound_name,
                    query=query,
                    mixed_audio=target_audio,
                    target_audio=target_audio,
                    context_desc="quiet",
                    snr_db=None
                )
                results['evaluations'].append(eval_result)
        
        else:
            # Test with background noise at different SNRs
            if noise_audio_paths:
                for noise_path in noise_audio_paths:
                    noise_audio, _ = self.load_audio(noise_path)
                    
                    for snr_db in context_config.get('snr_db', [0]):
                        # Mix audio
                        mixed_audio = self.mix_audio_signals(target_audio, noise_audio, snr_db)
                        
                        # Test each query variant
                        for query in sound_queries:
                            eval_result = self._evaluate_single_case(
                                sound_name=sound_name,
                                query=query,
                                mixed_audio=mixed_audio,
                                target_audio=target_audio,
                                context_desc=f"{context_name}_snr{snr_db}",
                                snr_db=snr_db,
                                noise_audio=noise_audio
                            )
                            results['evaluations'].append(eval_result)
        
        return results
    
    def _evaluate_single_case(self, sound_name: str, query: str,
                            mixed_audio: torch.Tensor, 
                            target_audio: torch.Tensor,
                            context_desc: str, snr_db: Optional[float],
                            noise_audio: Optional[torch.Tensor] = None) -> Dict:
        """Evaluate a single test case."""
        result = {
            'query': query,
            'context': context_desc,
            'snr_db': snr_db,
            'separation_metrics': {},
            'elimination_metrics': {}
        }
        
        # Save mixed audio
        mixed_path = self.mixed_audio_dir / f"{sound_name}_{context_desc}_{query.replace(' ', '_')}.wav"
        sf.write(mixed_path, mixed_audio.squeeze().cpu().numpy(), 32000)
        
        try:
            # Test separation (extract the triggering sound)
            separated_path = self.separated_audio_dir / f"{sound_name}_{context_desc}_{query.replace(' ', '_')}_separated.wav"
            separate_audio(
                model=self.model,
                audio_file=str(mixed_path),
                text=query,
                output_file=str(separated_path),
                device=self.device
            )
            
            # Load separated audio
            separated_audio, _ = self.load_audio(str(separated_path))
            
            # Calculate separation metrics
            is_quiet = context_desc == "quiet"
            result['separation_metrics'] = self.evaluate_separation(
                original=target_audio,
                separated=separated_audio,
                mixture=mixed_audio,
                is_quiet_context=is_quiet
            )
            
            # Test elimination (remove the triggering sound)
            eliminated_path = self.separated_audio_dir / f"{sound_name}_{context_desc}_{query.replace(' ', '_')}_eliminated.wav"
            remove_audio(
                model=self.model,
                audio_file=str(mixed_path),
                text=query,
                output_file=str(eliminated_path),
                device=self.device
            )
            
            # Load eliminated audio
            eliminated_audio, _ = self.load_audio(str(eliminated_path))
            
            # Calculate elimination metrics
            result['elimination_metrics'] = self.evaluate_elimination(
                original_mix=mixed_audio,
                eliminated=eliminated_audio,
                target_sound=target_audio
            )
            
            result['success'] = True
            
        except Exception as e:
            print(f"Error processing {sound_name} with query '{query}': {e}")
            result['success'] = False
            result['error'] = str(e)
        
        return result
    
    def run_full_evaluation(self, audio_samples: Dict[str, List[str]], 
                          noise_samples: Dict[str, List[str]]) -> None:
        """Run the complete evaluation across all sounds and contexts."""
        all_results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Progress tracking
        total_evaluations = len(TRIGGERING_SOUNDS) * len(AUDIO_CONTEXTS)
        pbar = tqdm(total=total_evaluations, desc="Evaluating sounds")
        
        for sound_name, sound_queries in TRIGGERING_SOUNDS.items():
            if sound_name not in audio_samples:
                print(f"Warning: No audio samples found for {sound_name}")
                continue
            
            sound_results = {
                'sound': sound_name,
                'contexts': {}
            }
            
            # Test in each context
            for context_name, context_config in AUDIO_CONTEXTS.items():
                # Get noise samples for this context
                noise_paths = None
                if context_config['noise_sounds']:
                    noise_paths = []
                    for noise_type in context_config['noise_sounds']:
                        if noise_type in noise_samples:
                            noise_paths.extend(noise_samples[noise_type][:2])  # Use first 2 samples
                
                # Process each audio sample
                context_results = []
                for audio_path in audio_samples[sound_name][:3]:  # Use first 3 samples
                    result = self.process_sound_in_context(
                        sound_name=sound_name,
                        sound_queries=sound_queries,
                        target_audio_path=audio_path,
                        context_name=context_name,
                        context_config=context_config,
                        noise_audio_paths=noise_paths
                    )
                    context_results.append(result)
                
                sound_results['contexts'][context_name] = context_results
                pbar.update(1)
            
            all_results.append(sound_results)
        
        pbar.close()
        
        # Save detailed results
        results_file = self.results_dir / f"evaluation_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Generate summary report
        self.generate_summary_report(all_results, timestamp)
        
        print(f"\nEvaluation complete! Results saved to {self.results_dir}")
    
    def generate_summary_report(self, results: List[Dict], timestamp: str) -> None:
        """Generate a summary report of the evaluation results."""
        summary = []
        
        for sound_result in results:
            sound_name = sound_result['sound']
            
            for context_name, context_results in sound_result['contexts'].items():
                # Aggregate metrics across all evaluations for this sound/context
                sep_metrics = []
                elim_metrics = []
                
                for result_group in context_results:
                    for eval in result_group['evaluations']:
                        if eval['success']:
                            sep_metrics.append(eval['separation_metrics'])
                            elim_metrics.append(eval['elimination_metrics'])
                
                if sep_metrics:
                    # Calculate average metrics
                    avg_sep = {
                        metric: np.mean([m[metric] for m in sep_metrics])
                        for metric in sep_metrics[0].keys()
                    }
                    avg_elim = {
                        metric: np.mean([m[metric] for m in elim_metrics])
                        for metric in elim_metrics[0].keys()
                    }
                    
                    summary.append({
                        'sound': sound_name,
                        'context': context_name,
                        'num_evaluations': len(sep_metrics),
                        'avg_separation_metrics': avg_sep,
                        'avg_elimination_metrics': avg_elim
                    })
        
        # Save summary as CSV
        summary_df = pd.DataFrame(summary)
        summary_file = self.results_dir / f"evaluation_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        
        # Print summary statistics
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        for context in AUDIO_CONTEXTS.keys():
            context_data = summary_df[summary_df['context'].str.contains(context)]
            if not context_data.empty:
                print(f"\n{context.upper()} CONTEXT:")
                print(f"Average SDR Improvement: {context_data['avg_separation_metrics'].apply(lambda x: x['sdr_improvement']).mean():.2f} dB")
                print(f"Average Target Residual: {context_data['avg_elimination_metrics'].apply(lambda x: x['target_residual_correlation']).mean():.3f}")
                print(f"Average Energy Reduction: {context_data['avg_elimination_metrics'].apply(lambda x: x['energy_reduction_ratio']).mean():.2%}")


def main():
    parser = argparse.ArgumentParser(description="AudioSep Noise Sensitivity Evaluation")
    parser.add_argument('--model-config', type=str, default='config/audiosep_base.yaml',
                      help='Path to model config file')
    parser.add_argument('--checkpoint', type=str, default='checkpoint/audiosep_base_4M_steps.ckpt',
                      help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='noise_sensitivity_results',
                      help='Output directory for results')
    parser.add_argument('--audio-samples-json', type=str, required=True,
                      help='JSON file mapping sound types to audio file paths')
    parser.add_argument('--noise-samples-json', type=str, required=True,
                      help='JSON file mapping noise types to audio file paths')
    
    args = parser.parse_args()
    
    # Load audio sample paths
    with open(args.audio_samples_json, 'r') as f:
        audio_samples = json.load(f)
    
    with open(args.noise_samples_json, 'r') as f:
        noise_samples = json.load(f)
    
    # Initialize evaluator
    evaluator = NoiseSensitivityEvaluator(
        model_config=args.model_config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir
    )
    
    # Run evaluation
    evaluator.run_full_evaluation(audio_samples, noise_samples)


if __name__ == "__main__":
    main()