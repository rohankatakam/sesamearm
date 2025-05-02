# SesameARM

Run Sesame Conversational Speech Model (CSM) on Apple Silicon Macs.

## Requirements

- Apple Silicon Mac (M1/M2/M3)
- macOS 12.0 or later
- Python 3.10+ 
- ffmpeg
- Hugging Face account with access to:
  - [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
  - [CSM-1B](https://huggingface.co/sesame/csm-1b)

## Setup

1. Clone the repository:
```bash
git clone https://github.com/rohankatakam/sesamearm.git
cd sesamearm
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
brew install ffmpeg
```

4. Set up Hugging Face authentication:
```bash
huggingface-cli login
cp .env.example .env
# Edit .env to add your Hugging Face token
```

## Running

Use the provided run script:
```bash
./run_on_mac.sh
```

This will:
- Set required environment variables for Apple Silicon
- Generate a sample conversation between two speakers
- Save the output as `full_conversation.wav` in the outputs directory

## Troubleshooting

- If you encounter MPS errors, ensure `PYTORCH_ENABLE_MPS_FALLBACK=1` is set in .env
- For model access issues, verify you've accepted model licenses on Hugging Face
- For persistent issues, try CPU mode: `python run_csm.py --device cpu`
