# CSM

**2025/03/13** - We are releasing the 1B CSM variant. The checkpoint is [hosted on Hugging Face](https://huggingface.co/sesame/csm_1b).

---

CSM (Conversational Speech Model) is a speech generation model from [Sesame](https://www.sesame.com) that generates RVQ audio codes from text and audio inputs. The model architecture employs a [Llama](https://www.llama.com/) backbone and a smaller audio decoder that produces [Mimi](https://huggingface.co/kyutai/mimi) audio codes.

A fine-tuned variant of CSM powers the [interactive voice demo](https://www.sesame.com/voicedemo) shown in our [blog post](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice).

A hosted [Hugging Face space](https://huggingface.co/spaces/sesame/csm-1b) is also available for testing audio generation.

## Requirements

* A CUDA-compatible GPU
* The code has been tested on CUDA 12.4 and 12.6, but it may also work on other versions
* Similarly, Python 3.10 is recommended, but newer versions may be fine
* For some audio operations, `ffmpeg` may be required
* Access to the following Hugging Face models:
  * [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
  * [CSM-1B](https://huggingface.co/sesame/csm-1b)

### Setup

```bash
git clone git@github.com:SesameAILabs/csm.git
cd csm
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Disable lazy compilation in Mimi
export NO_TORCH_COMPILE=1

# You will need access to CSM-1B and Llama-3.2-1B
huggingface-cli login
```

### Windows Setup

The `triton` package cannot be installed in Windows. Instead use `pip install triton-windows`.

## Quickstart

This script will generate a conversation between 2 characters, using a prompt for each character.

```bash
python run_csm.py
```

## Usage

If you want to write your own applications with CSM, the following examples show basic usage.

#### Generate a sentence

This will use a random speaker identity, as no prompt or context is provided.

```python
from generator import load_csm_1b
import torchaudio
import torch

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

generator = load_csm_1b(device=device)

audio = generator.generate(
    text="Hello from Sesame.",
    speaker=0,
    context=[],
    max_audio_length_ms=10_000,
)

torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
```

#### Generate with context

CSM sounds best when provided with context. You can prompt or provide context to the model using a `Segment` for each speaker's utterance.

NOTE: The following example is instructional and the audio files do not exist. It is intended as an example for using context with CSM.

```python
from generator import Segment

speakers = [0, 1, 0, 0]
transcripts = [
    "Hey how are you doing.",
    "Pretty good, pretty good.",
    "I'm great.",
    "So happy to be speaking to you.",
]
audio_paths = [
    "utterance_0.wav",
    "utterance_1.wav",
    "utterance_2.wav",
    "utterance_3.wav",
]

def load_audio(audio_path):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    return audio_tensor

segments = [
    Segment(text=transcript, speaker=speaker, audio=load_audio(audio_path))
    for transcript, speaker, audio_path in zip(transcripts, speakers, audio_paths)
]
audio = generator.generate(
    text="Me too, this is some cool stuff huh?",
    speaker=1,
    context=segments,
    max_audio_length_ms=10_000,
)

torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
```

## FAQ

**Does this model come with any voices?**

The model open-sourced here is a base generation model. It is capable of producing a variety of voices, but it has not been fine-tuned on any specific voice.

**Can I converse with the model?**

CSM is trained to be an audio generation model and not a general-purpose multimodal LLM. It cannot generate text. We suggest using a separate LLM for text generation.

**Does it support other languages?**

The model has some capacity for non-English languages due to data contamination in the training data, but it likely won't do well.

## Misuse and abuse ⚠️

This project provides a high-quality speech generation model for research and educational purposes. While we encourage responsible and ethical use, we **explicitly prohibit** the following:

- **Impersonation or Fraud**: Do not use this model to generate speech that mimics real individuals without their explicit consent.
- **Misinformation or Deception**: Do not use this model to create deceptive or misleading content, such as fake news or fraudulent calls.
- **Illegal or Harmful Activities**: Do not use this model for any illegal, harmful, or malicious purposes.

By using this model, you agree to comply with all applicable laws and ethical guidelines. We are **not responsible** for any misuse, and we strongly condemn unethical applications of this technology.

---

## Running on Apple Silicon (M1/M2/M3 Macs)

This fork includes modifications for running CSM on Apple Silicon Macs using Metal Performance Shaders (MPS). The following changes have been made:

1. Updated code to use MPS backend when available
2. Added environment variables for MPS support
3. Added helper scripts for setup and execution

### Requirements for Apple Silicon

* macOS on Apple Silicon (M1/M2/M3)
* Python 3.10+ (recommended)
* `ffmpeg` (install via `brew install ffmpeg`)

### Setup for Apple Silicon

```bash
# Clone the repository
git clone https://github.com/rohankatakam/sesamearm.git
cd sesamearm

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install ffmpeg if not already installed
brew install ffmpeg
```

### Environment Configuration

Create a `.env` file with your Hugging Face token:

```bash
# Hugging Face Authentication
HUGGING_FACE_TOKEN=your_huggingface_token_here

# Apple Silicon / MPS Support
NO_TORCH_COMPILE=1
PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Running on Apple Silicon

Use the provided run script:

```bash
./run_on_mac.sh
```

---

## Voice AI Bot using Pipecat and Sesame CSM

This project implements a voice AI bot that combines:
- Speech recognition (Whisper)
- LLM text generation (placeholder for Llama 3.2 1B)  
- Voice synthesis (Sesame CSM)

### Setup Instructions

1. Install system dependencies:
   ```bash
   brew install ffmpeg portaudio
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Implementations

This project provides two different implementations:

#### 1. Simple Voice Bot (`simple_voice_bot.py`)

A standalone implementation that integrates:
- Whisper for speech-to-text
- A simple pattern matcher (placeholder for Llama)
- Sesame CSM for voice synthesis

**Usage:**
```bash
python simple_voice_bot.py
```

This runs a simulation of the conversation flow with a predefined user message.

#### 2. Pipecat-based Voice Bot (`pipecat_voice_bot.py`)

A more complex implementation that uses the Pipecat framework for orchestrating the services:
- Whisper STT via Pipecat's services
- LLM text generation (pattern matcher)
- Sesame CSM for voice synthesis
- Audio output

**Usage:**
```bash
# Run with sample conversation
python pipecat_voice_bot.py --sample

# Run with custom text
python pipecat_voice_bot.py --text "Your custom message here"

# Process audio from a file (not fully implemented yet)
python pipecat_voice_bot.py --file /path/to/audio.wav
```

### Voice Bot Features

The voice AI bot implements realistic conversation behaviors:

1. **Turn-taking**: Waits for the user to finish speaking before responding

2. **Backchanneling**: The ability to insert conversational signals like "yeah", "I see" during pauses in user speech

3. **Interruptions**: Detect short pauses in user speech and politely interrupt when appropriate

4. **Natural Pauses**: Insert slight random pauses in generated speech for realism

### Generated Files

Audio outputs from the voice bot are saved to the `outputs` directory.

This script sets the necessary environment variables and runs the model with MPS acceleration.

### Troubleshooting

* If you encounter errors related to MPS operations, make sure `PYTORCH_ENABLE_MPS_FALLBACK=1` is set
* For any model download issues, verify you've accepted the license for each model on Hugging Face
* If you encounter any other issues, try running with CPU: `python run_csm.py --device cpu`

---

## Authors
Johan Schalkwyk, Ankit Kumar, Dan Lyth, Sefik Emre Eskimez, Zack Hodari, Cinjon Resnick, Ramon Sanabria, Raven Jiang, and the Sesame team.

Apple Silicon port by Rohan Katakam.
