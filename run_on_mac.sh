#!/bin/bash

# Script to run Sesame CSM-1B on macOS with Apple Silicon

# Set required environment variables
export NO_TORCH_COMPILE=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Activate virtual environment if not already activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    if [ -d "venv" ]; then
        echo "Activating virtual environment..."
        source venv/bin/activate
    else
        echo "Virtual environment not found. Please run setup first."
        exit 1
    fi
fi

# Create outputs directory if it doesn't exist
mkdir -p outputs

# Run the model
echo "Running Sesame CSM-1B with MPS acceleration..."
python run_csm.py

# Move the output to outputs directory
if [ -f "full_conversation.wav" ]; then
    echo "Moving output audio to outputs directory..."
    mv full_conversation.wav outputs/
    echo "✅ Audio generated successfully! Saved to outputs/full_conversation.wav"
else
    echo "❌ Error: No output audio file generated."
fi
