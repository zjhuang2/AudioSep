import numpy as np
import torch
import sounddevice as sd
import threading
import queue
from pipeline import build_audiosep
import time
from collections import deque
import scipy.signal as signal

class RealtimeSoundExtractor:
    def __init__(self, model, text_query, device='cuda', 
                 sample_rate=32000, chunk_duration=1.0, overlap=0.5):
        """
        Initialize real-time sound extractor
        
        Args:
            model: Pre-loaded AudioSep model
            text_query: Text description of sound to extract
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
        self.energy_ratio_history = deque(maxlen=50)
        
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
        """Process a single chunk of audio to extract target sound"""
        start_time = time.time()
        with torch.no_grad():
            # Prepare input
            audio_tensor = torch.Tensor(audio_chunk)[None, None, :].to(self.device)
            
            input_dict = {
                "mixture": audio_tensor,
                "condition": self.text_embedding,
            }
            
            # Get separated audio (the sound to extract)
            extracted_audio = self.model.ss_model(input_dict)["waveform"]
            extracted_audio = extracted_audio.squeeze(0).squeeze(0).cpu().numpy()
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(extracted_audio))
            if max_val > 0.9:
                extracted_audio = extracted_audio * 0.9 / max_val
            
            # Calculate quality metrics
            # SNR between extracted and original
            signal_power = np.mean(extracted_audio ** 2)
            noise_power = np.mean((audio_chunk - extracted_audio) ** 2)
            if noise_power > 0:
                snr_db = 10 * np.log10(signal_power / noise_power)
            else:
                snr_db = float('inf')
            
            # Energy ratio (how much energy was extracted)
            energy_ratio = signal_power / (np.mean(audio_chunk ** 2) + 1e-10)
            
            # Store metrics
            if not np.isnan(snr_db) and not np.isinf(snr_db):
                self.snr_history.append(snr_db)
            self.energy_ratio_history.append(energy_ratio)
            
            # Print metrics
            latency_ms = (time.time() - start_time) * 1000
            avg_snr = np.mean(list(self.snr_history)) if self.snr_history else 0
            avg_energy = np.mean(list(self.energy_ratio_history)) * 100
            
            print(f"\rLatency: {latency_ms:.1f}ms | SNR: {avg_snr:.1f}dB | Extracted: {avg_energy:.0f}%", 
                  end="", flush=True)
                
            return extracted_audio
            
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
        print(f"Starting real-time sound extractor...")
        print(f"Extracting sound: '{self.text_query}'")
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Chunk size: {self.chunk_size} samples ({self.chunk_size/self.sample_rate:.2f}s)")
        print(f"Overlap: {self.overlap_size} samples")
        print("Press Ctrl+C to stop")
        
        # Start processing thread
        self.stop_processing.clear()
        self.processing_thread = threading.Thread(target=self.processing_loop)
        self.processing_thread.start()
        
        # List available audio devices
        print("\nAvailable audio devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            print(f"{i}: {device['name']} - {device['max_input_channels']} in, {device['max_output_channels']} out")
        
        # Find laptop microphone (usually built-in)
        input_device = None
        output_device = None
        
        for i, device in enumerate(devices):
            device_name = device['name'].lower()
            # Look for built-in microphone
            if device['max_input_channels'] > 0 and ('built-in' in device_name or 'internal' in device_name or 'macbook' in device_name):
                if input_device is None:  # Use first match
                    input_device = i
            # Look for AirPods for output
            elif device['max_output_channels'] > 0 and 'airpods' in device_name:
                if output_device is None:  # Use first match
                    output_device = i
        
        # Fallback to default devices if not found
        if input_device is None:
            input_device = sd.default.device[0]
        if output_device is None:
            output_device = sd.default.device[1]
        
        print(f"\n=== Audio Device Configuration ===")
        print(f"INPUT:  Device #{input_device} - {devices[input_device]['name'] if input_device is not None else 'Default'}")
        print(f"OUTPUT: Device #{output_device} - {devices[output_device]['name'] if output_device is not None else 'Default'}")
        print(f"==================================\n")
        
        # Start audio stream with separate devices
        self.stream = sd.Stream(samplerate=self.sample_rate,
                               channels=1,
                               callback=self.audio_callback,
                               blocksize=self.hop_size,
                               dtype='float32',
                               device=(input_device, output_device))
        self.stream.start()
        
        # Keep running until stop is called
        try:
            while not self.stop_processing.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping sound extractor...")
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
        print("\nSound extractor stopped.")


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
    
    # Text query for sound to extract
    text_query = "human voice"  # Change this to extract specific sounds
    
    # Create and start real-time extractor
    extractor = RealtimeSoundExtractor(
        model=model,
        text_query=text_query,
        device=device,
        chunk_duration=2.0,  # Process 2-second chunks
        overlap=0.5  # 50% overlap
    )
    
    extractor.start()


if __name__ == "__main__":
    main()