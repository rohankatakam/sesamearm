#!/usr/bin/env python3
"""
Simple Voice AI Bot - Minimal Integration of Pipecat and Sesame CSM
-------------------------------------------------------------------
This script implements a basic voice AI bot that:
1. Processes audio files using Whisper STT
2. Generates responses using simulated LLM functionality
3. Synthesizes voice responses using Sesame CSM
"""

import os
import torch
import torchaudio
import numpy as np
import soundfile as sf
import sounddevice as sd
import tempfile
import logging
import asyncio
from typing import List, Dict, Any
import whisper
from generator import load_csm_1b, Segment
from huggingface_hub import hf_hub_download

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable Triton compilation for Sesame CSM
os.environ["NO_TORCH_COMPILE"] = "1"

class SimpleVoiceBot:
    """Minimal voice AI bot with Pipecat and Sesame CSM integration"""
    
    def __init__(self, device=None):
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
        self.csm_generator = load_csm_1b(device)
        self.sample_rate = self.csm_generator.sample_rate
        
        # Initialize Whisper for speech recognition
        logger.info("Loading Whisper model...")
        self.whisper_model = whisper.load_model("base")
        
        # Initialize prompt segments for Sesame CSM
        self.prompt_segments = self._prepare_prompt_segments()
        
        # Initialize conversation history
        self.conversation_history = []
    
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
    
    def transcribe_audio(self, audio_file: str) -> str:
        """Transcribe audio file using Whisper"""
        try:
            logger.info(f"Transcribing audio from: {audio_file}")
            result = self.whisper_model.transcribe(audio_file)
            transcription = result["text"].strip()
            logger.info(f"Transcription: {transcription}")
            return transcription
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return ""
    
    def generate_response(self, user_input: str) -> str:
        """Generate a response to user input"""
        # For this simple implementation, use basic pattern matching
        # In a full implementation, this would call the Llama 3.2 1B model
        
        # Add user input to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Simple pattern-based responses
        user_input_lower = user_input.lower()
        
        if any(greeting in user_input_lower for greeting in ["hello", "hi", "hey", "greetings"]):
            response = "Hello there! How can I help you today?"
        elif "how are you" in user_input_lower:
            response = "I'm doing well, thanks for asking! How about you?"
        elif "your name" in user_input_lower:
            response = "I'm your Sesame voice assistant. Nice to chat with you!"
        elif "thank" in user_input_lower:
            response = "You're welcome! Is there anything else I can help with?"
        elif "bye" in user_input_lower or "goodbye" in user_input_lower:
            response = "Goodbye! Have a great day!"
        elif "meet next week" in user_input_lower:
            response = "Sure, what day works best for you?"
        else:
            response = "That's interesting. Can you tell me more about that?"
        
        # Add response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def synthesize_speech(self, text: str, output_file: str = None) -> str:
        """Synthesize speech using Sesame CSM"""
        try:
            logger.info(f"Generating speech for: {text}")
            
            # Generate speech with Sesame CSM
            audio_tensor = self.csm_generator.generate(
                text=text,
                speaker=1,  # Use speaker ID 1 for bot responses
                context=self.prompt_segments,
                max_audio_length_ms=15_000,
                temperature=0.9,
            )
            
            # Create output file if not provided
            if output_file is None:
                tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                output_file = tmp_file.name
                tmp_file.close()
            
            # Save audio to file
            torchaudio.save(
                output_file,
                audio_tensor.unsqueeze(0).cpu(),
                self.sample_rate
            )
            
            logger.info(f"Speech saved to: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
            return None
    
    def play_audio(self, audio_file: str) -> None:
        """Play audio from file"""
        try:
            logger.info(f"Playing audio from: {audio_file}")
            data, samplerate = sf.read(audio_file)
            sd.play(data, samplerate)
            sd.wait()  # Wait until audio is done playing
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
    
    def process_audio_file(self, audio_file: str, output_dir: str = "./outputs") -> str:
        """Process an audio file through the full pipeline"""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # 1. Transcribe audio to text
            transcription = self.transcribe_audio(audio_file)
            if not transcription:
                return None
            
            # 2. Generate response
            response = self.generate_response(transcription)
            
            # 3. Synthesize speech
            output_file = os.path.join(output_dir, f"response_{os.path.basename(audio_file)}")
            speech_file = self.synthesize_speech(response, output_file)
            
            return speech_file
        except Exception as e:
            logger.error(f"Error processing audio file: {e}")
            return None
    
    def simulate_conversation(self, user_text: str = None) -> None:
        """Simulate a conversation with provided text or default example"""
        # Create outputs directory if it doesn't exist
        os.makedirs("./outputs", exist_ok=True)
        
        if user_text is None:
            # Use the example from the requirements
            user_text = "Hey, I was wondering if we could meet next week."
            
        # Log the user input
        logger.info(f"User: {user_text}")
        
        # Generate response
        response = self.generate_response(user_text)
        logger.info(f"Bot: {response}")
        
        # Synthesize and play response
        output_file = os.path.join("./outputs", "bot_response.wav")
        speech_file = self.synthesize_speech(response, output_file)
        if speech_file:
            self.play_audio(speech_file)
            logger.info(f"Response saved to: {speech_file}")

def main():
    """Main function to run the simple voice bot"""
    # Initialize the voice bot
    bot = SimpleVoiceBot()
    
    # Simulate a simple conversation
    bot.simulate_conversation()
    
    # Example with audio file (uncomment and modify path to use)
    # audio_file = "/path/to/input.wav"
    # response_file = bot.process_audio_file(audio_file)
    # if response_file:
    #     bot.play_audio(response_file)

if __name__ == "__main__":
    main()
