#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to authenticate with Hugging Face and download required models for Sesame CSM-1B.
"""

import os
import sys
import subprocess
from huggingface_hub import HfApi, login

def get_token_from_env():
    """Get Hugging Face token from .env file."""
    token = None
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                if line.startswith('HUGGING_FACE_TOKEN='):
                    token = line.strip().split('=', 1)[1]
                    # Remove quotes if present
                    token = token.strip('"\'')
                    break
    
    return token

def get_token_input():
    """Get Hugging Face token from user input."""
    print("\n" + "="*80)
    print("Hugging Face Authentication Required")
    print("="*80)
    print("To access the required models (Llama-3.2-1B and CSM-1B), you need to authenticate with Hugging Face.")
    print("Please visit: https://huggingface.co/settings/tokens to create a token if you don't have one.")
    print("\nEnter your Hugging Face token:")
    token = input().strip()
    return token

def authenticate_huggingface(token=None):
    """Authenticate with Hugging Face."""
    if not token:
        # First try to get token from .env file
        token = get_token_from_env()
        
        if token:
            print("Using token from .env file")
        else:
            # Fall back to user input if not in .env
            token = get_token_input()
    
    try:
        # Use the login function to authenticate
        login(token=token, write_permission=True)
        print("\n✅ Successfully authenticated with Hugging Face!")
        
        # Verify authentication by checking user info
        api = HfApi()
        user_info = api.whoami()
        print(f"Logged in as: {user_info['name']} ({user_info['email']})")
        return True
    except Exception as e:
        print(f"\n❌ Authentication failed: {str(e)}")
        print("Please try again with a valid token.")
        return False

def check_model_access():
    """Check access to the required models."""
    required_models = [
        "meta-llama/Llama-3.2-1B",
        "sesame/csm-1b"
    ]
    
    print("\nChecking access to required models...")
    api = HfApi()
    
    for model_id in required_models:
        try:
            # Try to get model info
            model_info = api.model_info(model_id)
            print(f"✅ You have access to {model_id}")
        except Exception as e:
            print(f"❌ Cannot access {model_id}: {str(e)}")
            print(f"Please make sure you have accepted the license for {model_id}")
            print(f"Visit https://huggingface.co/{model_id} to accept the license")
            return False
    
    return True

def create_output_dir():
    """Create output directory for generated audio."""
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\n✅ Created output directory at: {output_dir}")
    else:
        print(f"\n✅ Output directory already exists at: {output_dir}")

def main():
    # Authenticate with Hugging Face
    if not authenticate_huggingface():
        print("Authentication failed. Please run this script again with a valid token.")
        sys.exit(1)
    
    # Check model access
    if not check_model_access():
        print("\nYou need to accept the license for the required models.")
        print("Please visit the model pages on Hugging Face and accept the license agreements.")
        sys.exit(1)
    
    # Create output directory
    create_output_dir()
    
    print("\n" + "="*80)
    print("Setup completed successfully!")
    print("="*80)
    print("\nYou can now run the Sesame CSM-1B model with:")
    print("python run_csm.py")
    print("\nMake sure the environment variables are set:")
    print("export NO_TORCH_COMPILE=1")
    print("export PYTORCH_ENABLE_MPS_FALLBACK=1")

if __name__ == "__main__":
    main()
