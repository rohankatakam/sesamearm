#!/usr/bin/env python3
"""
Real-time voice recognition using microphone input and Whisper
"""
import sys
import time
import threading
import numpy as np
import torch
import whisper
import sounddevice as sd
import soundfile as sf
import argparse
from pathlib import Path

class MicrophoneToWhisper:
    """Records audio from microphone and transcribes it using Whisper in real-time"""
    
    def __init__(self, 
                 model_name="base",  # Use base model by default for better accuracy
                 device=None,
                 sample_rate=16000,
                 mic_device=None,
                 output_dir="outputs"):
        """Initialize the recorder and transcriber
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            device: Device to run Whisper on (cuda, cpu, mps)
            sample_rate: Audio sample rate
            mic_device: Index of microphone device to use
            output_dir: Directory to save audio files
        """
        # Set up device for Whisper
        if device is None:
            # Note: Force CPU on macOS due to Whisper compatibility issues with MPS
            if sys.platform == "darwin":
                device = "cpu"
                print(f"Using CPU on Apple Silicon (MPS not supported by Whisper)")
            elif torch.cuda.is_available():
                device = "cuda"
                print(f"Using CUDA for Whisper model")
            else:
                device = "cpu"
                print(f"Using CPU for Whisper model")
        
        self.device = device
        self.sample_rate = sample_rate
        self.mic_device = mic_device
        
        # Create output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Audio recording variables
        self.recording = False
        self.recorded_frames = []
        
        # VAD parameters
        self.speech_threshold = 6.0  # Even lower threshold for speech detection
        self.silence_threshold = 3.5  # Lower threshold for silence detection
        self.min_speech_frames = 8  # Slightly reduced to catch more speech segments
        self.min_silence_frames = 30  # Increased to ensure complete sentences
        self.speech_detected = False
        self.speech_frames = 0
        self.silence_frames = 0
        self.max_segment_frames = 2000  # Maximum segment length before forced segmentation
        
        # Whisper model
        print(f"Loading Whisper {model_name} model...")
        self.model = whisper.load_model(model_name, device=device)
        print("Whisper model loaded")
    
    def audio_callback(self, indata, frames, time, status):
        """Callback function for the InputStream"""
        if status:
            print(f"Status: {status}")
            
        # Store the audio data
        self.recorded_frames.append(indata.copy())
        
        # Calculate audio level for logging
        volume_norm = np.linalg.norm(indata) * 10
        sys.stdout.write(f"\rAudio level: {volume_norm:.2f}" + " " * 10)
        sys.stdout.flush()
        
        # Process voice activity detection
        self.process_vad(volume_norm)
    
    def process_vad(self, volume):
        """Process voice activity detection based on volume level"""
        if volume > self.speech_threshold:
            # This is speech
            self.speech_frames += 1
            self.silence_frames = 0
            
            if not self.speech_detected and self.speech_frames >= self.min_speech_frames:
                self.speech_detected = True
                print(f"\n🎤 Speech started (level: {volume:.2f})")
                
            # Handle extremely long segments by forcing segmentation
            if self.speech_detected and self.speech_frames >= self.max_segment_frames:
                print(f"\n⚠️ Maximum segment length reached ({self.speech_frames} frames)")
                threading.Thread(target=self.process_speech_segment).start()
                # Reset frame counter but keep speech_detected as True to continue recording
                self.speech_frames = 0
        else:
            # This might be silence
            if volume < self.silence_threshold:
                self.silence_frames += 1
            
            if self.speech_detected and self.silence_frames >= self.min_silence_frames:
                self.speech_detected = False
                print(f"\n🔇 Speech ended (after {self.speech_frames} frames)")
                
                # If we had a good amount of speech, transcribe it
                if self.speech_frames > 25:  # Require longer utterance for better transcription
                    # Process in a separate thread to not block audio recording
                    threading.Thread(target=self.process_speech_segment).start()
                else:
                    print(f"Speech segment too short ({self.speech_frames} frames), ignoring")
                
                # Reset counters
                self.speech_frames = 0
    
    def process_speech_segment(self):
        """Transcribe the current audio segment"""
        # Combine all recorded frames
        if not self.recorded_frames:
            return
            
        # Convert to numpy array
        audio_data = np.concatenate(self.recorded_frames, axis=0)
        
        # Reset recorded frames
        self.recorded_frames = []
        
        # Save audio file for debugging
        timestamp = int(time.time())
        audio_file = self.output_dir / f"speech_{timestamp}.wav"
        sf.write(str(audio_file), audio_data, self.sample_rate)
        print(f"Audio saved to {audio_file}")
        
        # Convert to float32 for Whisper with normalization and noise filtering
        audio_float = audio_data.astype(np.float32).reshape(-1) / 32768.0
        
        # Ensure proper audio level - normalize if too quiet
        max_amplitude = np.max(np.abs(audio_float))
        if max_amplitude < 0.1:  # If the audio is very quiet
            audio_float = audio_float * (0.5 / max_amplitude)  # Boost it, but not too much
        
        # Transcribe
        print("Transcribing...")
        try:
            result = self.model.transcribe(
                audio_float, 
                fp16=(self.device == 'cuda'),
                language="en",
                temperature=0.0,  # Use greedy decoding for more reliable results
                no_speech_threshold=0.6  # More aggressive speech detection
            )
            
            # Display result
            text = result["text"].strip()
            if text:
                print("\n" + "="*80)
                print(f"📝 TRANSCRIPTION: \"{text}\"")
                print("="*80 + "\n")
            else:
                print("No speech detected in the audio segment")
        except Exception as e:
            print(f"Error during transcription: {e}")
    
    def start_recording(self, duration=None):
        """Start recording audio
        
        Args:
            duration: If provided, record for this many seconds then stop
        """
        if self.recording:
            return
            
        self.recording = True
        self.recorded_frames = []
        
        # Get device info if needed
        if self.mic_device is not None:
            device_info = sd.query_devices(self.mic_device, 'input')
            print(f"Using microphone: {device_info['name']}")
        else:
            device_info = sd.query_devices(kind='input')
            print(f"Using default microphone: {device_info['name']}")
        
        # Create stream
        self.stream = sd.InputStream(
            device=self.mic_device,
            channels=1,
            samplerate=self.sample_rate,
            callback=self.audio_callback
        )
        
        # Start stream
        self.stream.start()
        print(f"Started recording" + (f" for {duration} seconds" if duration else ""))
        
        # If duration provided, stop after that time
        if duration:
            time.sleep(duration)
            self.stop_recording()
    
    def stop_recording(self):
        """Stop recording audio"""
        if not self.recording:
            return
            
        self.recording = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        print("\nRecording stopped")
        
        # Process any remaining speech
        if self.speech_detected and self.speech_frames > 15:
            self.process_speech_segment()


def list_audio_devices():
    """List available audio input devices"""
    print("\n===== AVAILABLE MICROPHONES =====")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        input_channels = device.get('max_input_channels', 0)
        if input_channels > 0:  # Only show input devices
            print(f"Device {i}: {device['name']} (Inputs: {input_channels})")
    print("=================================\n")


def main():
    parser = argparse.ArgumentParser(description="Real-time Voice Recognition")
    parser.add_argument("--model", type=str, default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (base recommended for balance of speed/accuracy)")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cuda", "cpu", "mps"],
                        help="Compute device (automatically detected if not specified)")
    parser.add_argument("--mic-device", type=int, default=None,
                        help="Microphone device index (use --list-devices to see options)")
    parser.add_argument("--list-devices", action="store_true",
                        help="List available microphones and exit")
    parser.add_argument("--duration", type=int, default=None,
                        help="Recording duration in seconds (default: continuous until Ctrl+C)")
    args = parser.parse_args()
    
    if args.list_devices:
        list_audio_devices()
        return
    
    # Create voice recognizer
    recognizer = MicrophoneToWhisper(
        model_name=args.model,
        device=args.device,
        mic_device=args.mic_device
    )
    
    try:
        print("\nStarting voice recognition. Speak into your microphone.")
        print("Press Ctrl+C to stop\n")
        
        # Start recording
        recognizer.start_recording(args.duration)
        
        # If no duration set, keep running until interrupted
        if not args.duration:
            while True:
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        # Stop recording
        recognizer.stop_recording()


if __name__ == "__main__":
    main()
