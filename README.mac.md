# Sesame CSM-1B on Apple Silicon (M1/M2/M3)

This guide provides instructions for running the Sesame CSM-1B model on Apple Silicon Macs.

## Setup

We've modified the repository to work with the Metal Performance Shaders (MPS) backend on Apple Silicon. Follow these steps to set up and run the model:

### Prerequisites

- macOS on Apple Silicon (M1/M2/M3)
- Python 3.10+ (recommended)
- Hugging Face account with access to required models

### Installation

1. **Environment Variables**

   Set the required environment variables:

   ```bash
   export NO_TORCH_COMPILE=1
   export PYTORCH_ENABLE_MPS_FALLBACK=1
   ```

2. **Virtual Environment**

   Create and activate a Python virtual environment:

   ```bash
   python3.10 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Hugging Face Authentication**

   Run the setup script:

   ```bash
   python setup_huggingface.py
   ```

   Follow the prompts to enter your Hugging Face token. You'll need access to:
   - `meta-llama/Llama-3.2-1B`
   - `sesame/csm-1b`

## Running the Model

Generate audio using the example script:

```bash
python run_csm.py
```

The generated audio will be saved as `full_conversation.wav` in the current directory.

## Troubleshooting

### MPS Fallback Issues

If you encounter errors related to MPS operations, try:

1. Make sure `PYTORCH_ENABLE_MPS_FALLBACK=1` is set
2. Run with CPU backend if MPS fails: `python run_csm.py --device cpu`

### Model Download Issues

If you have issues downloading models:

1. Check your Hugging Face token
2. Make sure you've accepted the license for each model on Hugging Face
3. Run `huggingface-cli login` manually if needed

### bfloat16 Support

If you encounter errors related to bfloat16 dtype, modify `generator.py` to use float32 instead.

## Notes

- Performance may vary based on your Mac's hardware
- The first run will download the models, which may take some time
- Audio generation may be slower on MPS than on CUDA GPUs
