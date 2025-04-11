#!/usr/bin/env python3
"""
Pipecat Voice AI Bot
-------------------
This implementation uses Pipecat's architecture to create a conversational AI bot:
1. STT (Speech-to-Text) using Whisper via Pipecat
2. LLM processing with Llama 3.2 1B (simulated)
3. TTS (Text-to-Speech) using Sesame CSM
"""

import os
import sys
import asyncio
import logging
import tempfile
import numpy as np
import torch
import torchaudio
import soundfile as sf
import sounddevice as sd
import time
from typing import Dict, Any, Optional, List, Tuple

# Pipecat imports
from pipecat.frames.frames import (
    InputAudioRawFrame,
    OutputAudioRawFrame,
    AudioRawFrame,
    TextFrame,
    TranscriptionFrame
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameProcessor
from pipecat import connect
from pipecat.services.whisper import WhisperSTTService, Model

# Sesame CSM imports
from generator import load_csm_1b, Segment
from huggingface_hub import hf_hub_download

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable Triton compilation for Sesame CSM
os.environ["NO_TORCH_COMPILE"] = "1"

class SesameCSMTTS(FrameProcessor):
    """Custom Pipecat-compatible TTS service using Sesame CSM"""
    
    def __init__(self, device=None):
        super().__init__()
        # Select best available device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
                logger.info("Using MPS (Metal Performance Shaders) for Apple Silicon")
            else:
                device = "cpu"
                logger.info("CUDA and MPS not available, falling back to CPU")
        
        self.device = device
        logger.info(f"Using device: {device}")
        
        # Load Sesame CSM model
        logger.info("Loading Sesame CSM model...")
        self.csm_generator = load_csm_1b(device)
        self.sample_rate = self.csm_generator.sample_rate
        
        # Initialize prompt segments for Sesame CSM
        self.prompt_segments = self._prepare_prompt_segments()
        
        # Create outputs directory
        os.makedirs("./outputs", exist_ok=True)
    
    def _prepare_prompt_segments(self) -> List[Segment]:
        """Prepare the prompt segments for Sesame CSM"""
        # Download prompt files if not already cached
        prompt_a_path = hf_hub_download(
            repo_id="sesame/csm-1b",
            filename="prompts/conversational_a.wav"
        )
        prompt_b_path = hf_hub_download(
            repo_id="sesame/csm-1b",
            filename="prompts/conversational_b.wav"
        )
        
        prompt_texts = {
            "conversational_a": (
                "like revising for an exam I'd have to try and like keep up the momentum because I'd "
                "start really early I'd be like okay I'm gonna start revising now and then like "
                "you're revising for ages and then I just like start losing steam I didn't do that "
                "for the exam we had recently to be fair that was a more of a last minute scenario "
                "but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I "
                "sort of start the day with this not like a panic but like a"
            ),
            "conversational_b": (
                "like a super Mario level. Like it's very like high detail. And like, once you get "
                "into the park, it just like, everything looks like a computer game and they have all "
                "these, like, you know, if, if there's like a, you know, like in a Mario game, they "
                "will have like a question block. And if you like, you know, punch it, a coin will "
                "come out. So like everyone, when they come into the park, they get like this little "
                "bracelet and then you can go punching question blocks around."
            )
        }
        
        # Load audio prompts
        def load_prompt_audio(audio_path, target_sample_rate):
            audio_tensor, sample_rate = torchaudio.load(audio_path)
            audio_tensor = audio_tensor.squeeze(0)
            # Resample is lazy so we can always call it
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, orig_freq=sample_rate, new_freq=target_sample_rate
            )
            return audio_tensor
        
        # Prepare prompt segments
        prompt_a = Segment(
            text=prompt_texts["conversational_a"],
            speaker=0,
            audio=load_prompt_audio(prompt_a_path, self.sample_rate)
        )
        
        prompt_b = Segment(
            text=prompt_texts["conversational_b"],
            speaker=1,
            audio=load_prompt_audio(prompt_b_path, self.sample_rate)
        )
        
        return [prompt_a, prompt_b]
    
    async def process(self, frame):
        """Process a text frame and convert it to audio using Sesame CSM"""
        if not isinstance(frame, TextFrame):
            # Pass through frames we don't handle
            return frame
            
        try:
            text = frame.text
            logger.info(f"Generating speech for: {text}")
            
            # Generate speech with Sesame CSM
            audio_tensor = self.csm_generator.generate(
                text=text,
                speaker=1,  # Use speaker ID 1 for bot responses
                context=self.prompt_segments,
                max_audio_length_ms=15_000,
                temperature=0.9,
            )
            
            # Save audio to file for debugging/reference
            output_file = os.path.join("./outputs", "bot_response.wav")
            torchaudio.save(
                output_file,
                audio_tensor.unsqueeze(0).cpu(),
                self.sample_rate
            )
            logger.info(f"Speech saved to: {output_file}")
            
            # Convert audio tensor to bytes for AudioRawFrame
            audio_bytes = audio_tensor.cpu().numpy().tobytes()
            
            # Create and return OutputAudioRawFrame
            return OutputAudioRawFrame(
                audio=audio_bytes,
                sample_rate=self.sample_rate,
                num_channels=1
            )
            
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            # Pass through the original frame on error
            return frame

class SimpleLlama(FrameProcessor):
    """Simple LLM service using pattern matching (placeholder for Llama 3.2 1B)"""
    
    def __init__(self):
        super().__init__()
        self.conversation_history = []
    
    async def process(self, frame):
        """Process a text frame and generate a response"""
        # If the frame is a TranscriptionFrame, extract the text from it
        if isinstance(frame, TranscriptionFrame):
            text = frame.text
        # If it's a TextFrame, use it directly
        elif isinstance(frame, TextFrame):
            text = frame.text
        else:
            # Pass through frames we don't handle
            return frame
        
        try:
            logger.info(f"User: {text}")
            
            # Add user input to conversation history
            self.conversation_history.append({"role": "user", "content": text})
            
            # Simple pattern-based responses
            # In a real implementation, this would call the Llama 3.2 1B model
            text_lower = text.lower()
            
            if any(greeting in text_lower for greeting in ["hello", "hi", "hey", "greetings"]):
                response = "Hello there! How can I help you today?"
            elif "how are you" in text_lower:
                response = "I'm doing well, thanks for asking! How about you?"
            elif "your name" in text_lower:
                response = "I'm your Sesame voice assistant. Nice to chat with you!"
            elif "thank" in text_lower:
                response = "You're welcome! Is there anything else I can help with?"
            elif "bye" in text_lower or "goodbye" in text_lower:
                response = "Goodbye! Have a great day!"
            elif "meet next week" in text_lower:
                response = "Sure, what day works best for you?"
            else:
                response = "That's interesting. Can you tell me more about that?"
            
            # Add response to conversation history
            self.conversation_history.append({"role": "assistant", "content": response})
            
            logger.info(f"Bot: {response}")
            return TextFrame(response)
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return TextFrame("I'm sorry, I couldn't process your request.")

class AudioOutput(FrameProcessor):
    """Service to output audio through speakers"""
    
    def __init__(self):
        super().__init__()
    
    async def process(self, frame):
        """Play the audio from the frame"""
        if not isinstance(frame, OutputAudioRawFrame) and not isinstance(frame, AudioRawFrame):
            # Pass through frames we don't handle
            return frame
            
        try:
            # Get audio data from frame
            audio_bytes = frame.audio
            sample_rate = frame.sample_rate
            num_channels = frame.num_channels
            
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32).reshape(-1, num_channels)
            
            # Create a temporary file to save the audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_filename = temp_file.name
            
            # Save to temporary file
            sf.write(temp_filename, audio_array, sample_rate)
            
            # Play the audio
            logger.info(f"Playing audio...")
            data, sr = sf.read(temp_filename)
            sd.play(data, sr)
            sd.wait()  # Wait until audio is done playing
            
            # Clean up
            os.unlink(temp_filename)
            
            # Pass through the frame
            return frame
                
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
            return frame

class AudioFileInput:
    """Class to input audio from a file"""
    
    def __init__(self, audio_file_path: str):
        self.audio_file_path = audio_file_path
    
    async def read_audio(self) -> InputAudioRawFrame:
        """Read the audio file and return an InputAudioRawFrame"""
        try:
            logger.info(f"Reading audio file: {self.audio_file_path}")
            
            # Read audio file
            data, sample_rate = sf.read(self.audio_file_path)
            
            # Convert to mono if needed
            if len(data.shape) > 1 and data.shape[1] > 1:
                data = data[:, 0]
                num_channels = 1
            else:
                num_channels = 1
            
            # Convert to bytes
            audio_bytes = data.astype(np.float32).tobytes()
            
            # Create and return InputAudioRawFrame
            return InputAudioRawFrame(
                audio=audio_bytes,
                sample_rate=sample_rate,
                num_channels=num_channels
            )
            
        except Exception as e:
            logger.error(f"Error reading audio file: {e}")
            # Return empty audio frame on error
            empty_audio = np.zeros(1000, dtype=np.float32).tobytes()
            return InputAudioRawFrame(
                audio=empty_audio,
                sample_rate=16000,
                num_channels=1
            )

class TextInput:
    """Class to simulate text input"""
    
    def __init__(self, text: str):
        self.text = text
    
    async def get_text(self) -> TextFrame:
        """Return a TextFrame with the simulated message"""
        logger.info(f"Simulating user input: {self.text}")
        return TextFrame(self.text)

async def process_audio_file(file_path: str):
    """Process audio from a file through the full pipeline"""
    logger.info("Setting up pipeline for audio file processing...")
    
    try:
        # Select best available device
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using MPS (Metal Performance Shaders) for Apple Silicon")
        else:
            device = "cpu"
            logger.info("Using CPU for processing")
            
        # Create input, processors, and pipeline
        audio_input = AudioFileInput(file_path)
        whisper = WhisperSTTService(model=Model.TINY, device=device)
        llm = SimpleLlama()
        tts = SesameCSMTTS(device=device)
        audio_output = AudioOutput()
        
        # Build the pipeline with processors
        processors = [whisper, llm, tts, audio_output]
        pipeline = Pipeline(processors)
        
        # Create pipeline runner and task
        runner = PipelineRunner()
        task = PipelineTask(pipeline)
        
        # Get audio frame from file
        audio_frame = await audio_input.read_audio()
        
        # Queue the audio frame for processing
        await task.queue_frame(audio_frame)
        
        # Run the pipeline task
        await runner.run(task)
        
        logger.info("Audio file processing complete")
        
    except Exception as e:
        logger.error(f"Error in pipeline: {e}")

async def simulate_conversation(text: str = "Hey, I was wondering if we could meet next week."):
    """Simulate a conversation with provided text"""
    logger.info("Setting up pipeline for simulated conversation...")
    
    try:
        # Select best available device
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using MPS (Metal Performance Shaders) for Apple Silicon")
        else:
            device = "cpu"
            logger.info("Using CPU for processing")
            
        # Create input, processors, and pipeline
        text_input = TextInput(text)
        llm = SimpleLlama()
        tts = SesameCSMTTS(device=device)
        audio_output = AudioOutput()
        
        # Build the pipeline with processors
        processors = [llm, tts, audio_output]
        pipeline = Pipeline(processors)
        
        # Create pipeline runner and task
        runner = PipelineRunner()
        task = PipelineTask(pipeline)
        
        # Get text frame from simulated input
        text_frame = await text_input.get_text()
        
        # Queue the text frame for processing
        await task.queue_frame(text_frame)
        
        # Run the pipeline task
        await runner.run(task)
        
        logger.info("Conversation simulation complete")
        
    except Exception as e:
        logger.error(f"Error in pipeline: {e}")

async def main():
    """Main function to run the voice AI bot"""
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Pipecat Voice AI Bot')
    parser.add_argument('--file', '-f', type=str, help='Path to audio file for input')
    parser.add_argument('--text', '-t', type=str, help='Text input for simulation')
    parser.add_argument('--sample', '-s', action='store_true', help='Use a sample conversation')
    args = parser.parse_args()
    
    # Create outputs directory
    os.makedirs("./outputs", exist_ok=True)
    
    if args.file:
        # Process audio from file
        await process_audio_file(args.file)
    elif args.sample:
        # Run the sample conversation that matches the requirements
        await simulate_conversation("Hey, I was wondering if we could meet next week.")
    else:
        # Simulate conversation with provided text or default
        text = args.text if args.text else "Hello, how are you today?"
        await simulate_conversation(text)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
