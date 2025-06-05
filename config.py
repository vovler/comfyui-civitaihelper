"""
Configuration settings for Civitai Helper ComfyUI Extension
"""

# Civitai API Configuration
CIVITAI_API_BASE_URL = "https://civitai.com/api/v1"
CIVITAI_SEARCH_ENDPOINT = f"{CIVITAI_API_BASE_URL}/models"
CIVITAI_MODEL_DETAIL_ENDPOINT = f"{CIVITAI_API_BASE_URL}/models"

# Model type mappings for ComfyUI folder structure
MODEL_TYPE_MAPPING = {
    'Checkpoint': 'checkpoints',
    'LORA': 'loras', 
    'LoCon': 'loras',
    'VAE': 'vae',
    'TextualInversion': 'embeddings',
    'ControlNet': 'controlnet',
    'CLIP': 'clip_vision',
    'Upscaler': 'upscale_models'
}

# File extensions by model type
MODEL_EXTENSIONS = {
    'checkpoints': ['.safetensors', '.ckpt', '.pt'],
    'loras': ['.safetensors', '.pt'],
    'vae': ['.safetensors', '.pt', '.ckpt'],
    'embeddings': ['.safetensors', '.pt', '.bin'],
    'controlnet': ['.safetensors', '.pt'],
    'clip_vision': ['.safetensors', '.pt'],
    'upscale_models': ['.pth', '.pt', '.safetensors']
}

# Download settings
DEFAULT_DOWNLOAD_CHUNK_SIZE = 8192
PROGRESS_LOG_INTERVAL = 1024 * 1024 * 10  # Log every 10MB

# PNG metadata fields to check for workflow data
WORKFLOW_METADATA_FIELDS = [
    'workflow',
    'Workflow', 
    'comfyui_workflow',
    'ComfyUI',
    'parameters',
    'Parameters'
]

# Model search settings
MAX_SEARCH_RESULTS = 10
MODEL_MATCH_THRESHOLD = 0.8 