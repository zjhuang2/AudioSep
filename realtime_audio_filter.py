import numpy as np
import torch
import sounddevice as sd
import threading
import queue
from pipeline import build_audiosep
import time
from collections import deque
import scipy.signal as signal

class RealtimeAudioFilter:
    def __init__(self, model, text_query, device='cuda', 
                 sample_rate=32000, chunk_duration=1.0, overlap=0.5):
        """
        Initialize real-time audio filter
        
        Args:
            model: Pre-loaded AudioSep model
            text_query: Text description of sound to remove
            device: PyTorch device
            sample_rate: Sample rate (AudioSep uses 32kHz)
            chunk_duration: Duration of each processing chunk in seconds
            overlap: Overlap between chunks (0-1)
        """
        self.model = model
        self.text_query = text_query
        self.device = device
        self.sample_rate = sample_rate
        self.chunk_size = int(chunk_duration * sample_rate)
        self.overlap_size = int(self.chunk_size * overlap)
        self.hop_size = self.chunk_size - self.overlap_size
        
        # Get text embedding once
        with torch.no_grad():
            self.text_embedding = model.query_encoder.get_query_embed(
                modality='text',
                text=[text_query],
                device=device
            )
        
        # Buffers for audio processing
        self.input_buffer = deque(maxlen=self.chunk_size)
        self.output_buffer = deque(maxlen=self.chunk_size * 2)
        self.overlap_buffer = np.zeros(self.overlap_size)
        
        # Thread-safe queues
        self.input_queue = queue.Queue(maxsize=10)
        self.output_queue = queue.Queue(maxsize=10)
        
        # Processing thread
        self.processing_thread = None
        self.stop_processing = threading.Event()
        
        # Window for smooth transitions
        self.window = signal.windows.hann(self.chunk_size)
        
        # Quality metrics
        self.snr_history = deque(maxlen=50)
        self.suppression_history = deque(maxlen=50)
        
    def audio_callback(self, indata, outdata, frames, time, status):
        """Callback for sounddevice stream"""
        if status:
            print(f"Audio callback status: {status}")
        
        # Add input to queue for processing
        self.input_queue.put(indata.copy())
        
        # Get processed output if available
        try:
            processed = self.output_queue.get_nowait()
            outdata[:] = processed.reshape(-1, 1)
        except queue.Empty:
            # If no processed audio ready, output silence
            outdata[:] = 0
            
    def process_audio_chunk(self, audio_chunk):
        """Process a single chunk of audio to remove target sound"""
        start_time = time.time()
        with torch.no_grad():
            # Prepare input
            audio_tensor = torch.Tensor(audio_chunk)[None, None, :].to(self.device)
            
            input_dict = {
                "mixture": audio_tensor,
                "condition": self.text_embedding,
            }
            
            # Get separated audio (the sound to remove)
            sep_audio = self.model.ss_model(input_dict)["waveform"]
            sep_audio = sep_audio.squeeze(0).squeeze(0).cpu().numpy()
            
            # Subtract from original to remove the sound
            filtered_audio = audio_chunk - sep_audio
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(filtered_audio))
            if max_val > 0.9:
                filtered_audio = filtered_audio * 0.9 / max_val
            
            # Calculate quality metrics
            # SNR of filtered output
            signal_power = np.mean(filtered_audio ** 2)
            removed_power = np.mean(sep_audio ** 2)
            if removed_power > 0:
                snr_db = 10 * np.log10(signal_power / removed_power)
            else:
                snr_db = float('inf')
            
            # Suppression ratio (how much was removed)
            suppression_ratio = removed_power / (np.mean(audio_chunk ** 2) + 1e-10)
            
            # Store metrics
            if not np.isnan(snr_db) and not np.isinf(snr_db):
                self.snr_history.append(snr_db)
            self.suppression_history.append(suppression_ratio)
            
            # Print metrics
            latency_ms = (time.time() - start_time) * 1000
            avg_snr = np.mean(list(self.snr_history)) if self.snr_history else 0
            avg_suppression = np.mean(list(self.suppression_history)) * 100
            
            print(f"\rLatency: {latency_ms:.1f}ms | SNR: {avg_snr:.1f}dB | Removed: {avg_suppression:.0f}%", 
                  end="", flush=True)
                
            return filtered_audio
            
    def processing_loop(self):
        """Main processing thread"""
        accumulated_input = np.array([])
        
        while not self.stop_processing.is_set():
            try:
                # Get input audio
                input_chunk = self.input_queue.get(timeout=0.1)
                input_audio = input_chunk.flatten()
                
                # Accumulate input
                accumulated_input = np.concatenate([accumulated_input, input_audio])
                
                # Process when we have enough samples
                while len(accumulated_input) >= self.chunk_size:
                    # Extract chunk
                    chunk = accumulated_input[:self.chunk_size]
                    
                    # Process chunk
                    processed = self.process_audio_chunk(chunk)
                    
                    # Apply window for smooth transitions
                    processed = processed * self.window
                    
                    # Handle overlap-add
                    output = np.zeros(self.chunk_size)
                    output[:self.overlap_size] = self.overlap_buffer + processed[:self.overlap_size]
                    output[self.overlap_size:] = processed[self.overlap_size:]
                    
                    # Save overlap for next chunk
                    self.overlap_buffer = processed[-self.overlap_size:]
                    
                    # Output the hop_size samples
                    self.output_queue.put(output[:self.hop_size])
                    
                    # Move to next chunk
                    accumulated_input = accumulated_input[self.hop_size:]
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                
    def start(self):
        """Start real-time audio processing"""
        print(f"Starting real-time audio filter...")
        print(f"Removing sound: '{self.text_query}'")
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Chunk size: {self.chunk_size} samples ({self.chunk_size/self.sample_rate:.2f}s)")
        print(f"Overlap: {self.overlap_size} samples")
        print("Press Ctrl+C to stop")
        
        # Start processing thread
        self.stop_processing.clear()
        self.processing_thread = threading.Thread(target=self.processing_loop)
        self.processing_thread.start()
        
        # Start audio stream
        self.stream = sd.Stream(samplerate=self.sample_rate,
                               channels=1,
                               callback=self.audio_callback,
                               blocksize=self.hop_size,
                               dtype='float32')
        self.stream.start()
        
        # Keep running until stop is called
        try:
            while not self.stop_processing.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping audio filter...")
        finally:
            self.stop()
            
    def stop(self):
        """Stop audio processing"""
        self.stop_processing.set()
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop()
            self.stream.close()
        if self.processing_thread:
            self.processing_thread.join()
        # Clear queues
        while not self.input_queue.empty():
            self.input_queue.get()
        while not self.output_queue.empty():
            self.output_queue.get()
        print("\nAudio filter stopped.")


def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading AudioSep model...")
    model = build_audiosep(
        config_yaml='config/audiosep_base.yaml',
        checkpoint_path='checkpoint/audiosep_base_4M_steps.ckpt',
        device=device
    )
    
    # Text query for sound to remove
    text_query = "background noise"  # Change this to target specific sounds
    
    # Create and start real-time filter
    filter = RealtimeAudioFilter(
        model=model,
        text_query=text_query,
        device=device,
        chunk_duration=2.0,  # Process 2-second chunks
        overlap=0.5  # 50% overlap
    )
    
    filter.start()


if __name__ == "__main__":
    main()