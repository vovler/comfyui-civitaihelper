"""
Utility functions for Civitai Helper ComfyUI Extension
"""

import os
import json
import hashlib
import difflib
from typing import Dict, List, Optional, Tuple, Any
from .config import MODEL_EXTENSIONS, WORKFLOW_METADATA_FIELDS


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing/replacing invalid characters
    """
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255-len(ext)] + ext
    
    return filename


def calculate_file_hash(file_path: str, algorithm: str = 'sha256') -> str:
    """
    Calculate hash of a file
    """
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def find_models_in_workflow(workflow: Dict) -> List[Dict[str, Any]]:
    """
    Extract all model references from a ComfyUI workflow
    """
    models = []
    
    if not isinstance(workflow, dict):
        return models
    
    # Handle different workflow formats
    nodes = []
    
    # Standard workflow format
    if 'nodes' in workflow:
        nodes = workflow['nodes']
    # API format
    elif isinstance(workflow, dict) and all(k.isdigit() for k in workflow.keys()):
        nodes = [{'id': k, 'inputs': v.get('inputs', {}), 'class_type': v.get('class_type', '')} 
                for k, v in workflow.items()]
    
    # Extract models from nodes
    for node in nodes:
        if not isinstance(node, dict) or 'inputs' not in node:
            continue
            
        inputs = node['inputs']
        node_id = node.get('id', 'unknown')
        class_type = node.get('class_type', '')
        
        # Check for different model input types
        model_inputs = {
            'ckpt_name': 'checkpoints',
            'checkpoint': 'checkpoints',
            'model_name': 'checkpoints',
            'lora_name': 'loras',
            'lora': 'loras',
            'vae_name': 'vae',
            'vae': 'vae',
            'embedding_name': 'embeddings',
            'embedding': 'embeddings',
            'control_net_name': 'controlnet',
            'controlnet': 'controlnet',
            'clip_name': 'clip_vision',
            'upscale_model': 'upscale_models'
        }
        
        for input_key, model_type in model_inputs.items():
            if input_key in inputs and inputs[input_key]:
                model_name = inputs[input_key]
                if isinstance(model_name, str) and model_name.strip():
                    models.append({
                        'name': model_name.strip(),
                        'type': model_type,
                        'node_id': node_id,
                        'class_type': class_type,
                        'input_key': input_key
                    })
    
    return models


def normalize_model_name(name: str) -> str:
    """
    Normalize model name for comparison
    """
    # Remove file extension
    name = os.path.splitext(name)[0]
    
    # Convert to lowercase
    name = name.lower()
    
    # Remove common suffixes
    suffixes_to_remove = [
        '_v1', '_v2', '_v3', '_v4', '_v5',
        '_fp16', '_fp32', '_pruned', '_ema',
        '_inpainting', '_768', '_512', '_1024',
        '_safetensors', '_pytorch'
    ]
    
    for suffix in suffixes_to_remove:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    
    # Replace common separators with spaces
    name = name.replace('_', ' ').replace('-', ' ')
    
    # Remove extra spaces
    name = ' '.join(name.split())
    
    return name


def fuzzy_match_model_name(search_name: str, candidate_name: str, threshold: float = 0.6) -> bool:
    """
    Perform fuzzy matching between model names
    """
    search_normalized = normalize_model_name(search_name)
    candidate_normalized = normalize_model_name(candidate_name)
    
    # Exact match
    if search_normalized == candidate_normalized:
        return True
    
    # Substring match
    if search_normalized in candidate_normalized or candidate_normalized in search_normalized:
        return True
    
    # Fuzzy match using difflib
    similarity = difflib.SequenceMatcher(None, search_normalized, candidate_normalized).ratio()
    return similarity >= threshold


def get_model_file_size(file_path: str) -> int:
    """
    Get the size of a model file in bytes
    """
    try:
        return os.path.getsize(file_path)
    except OSError:
        return 0


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024.0 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def validate_workflow_json(workflow_data: str) -> Optional[Dict]:
    """
    Validate and parse workflow JSON data
    """
    try:
        workflow = json.loads(workflow_data)
        
        # Basic validation
        if not isinstance(workflow, dict):
            return None
            
        # Check if it looks like a ComfyUI workflow
        has_nodes = 'nodes' in workflow
        has_api_format = isinstance(workflow, dict) and any(k.isdigit() for k in workflow.keys())
        
        if not (has_nodes or has_api_format):
            return None
            
        return workflow
        
    except (json.JSONDecodeError, TypeError):
        return None


def find_existing_model_file(model_name: str, model_dir: str, model_type: str) -> Optional[str]:
    """
    Find an existing model file in the specified directory
    """
    if not os.path.exists(model_dir):
        return None
    
    # Get valid extensions for this model type
    valid_extensions = MODEL_EXTENSIONS.get(model_type, ['.safetensors', '.pt', '.ckpt'])
    
    # Try exact filename matches first
    for ext in valid_extensions:
        # Try with original name
        full_path = os.path.join(model_dir, model_name)
        if not full_path.endswith(ext):
            full_path = f"{os.path.splitext(full_path)[0]}{ext}"
        
        if os.path.exists(full_path):
            return full_path
    
    # Try fuzzy matching with existing files
    try:
        existing_files = [f for f in os.listdir(model_dir) 
                         if any(f.lower().endswith(ext) for ext in valid_extensions)]
        
        for existing_file in existing_files:
            if fuzzy_match_model_name(model_name, existing_file):
                return os.path.join(model_dir, existing_file)
                
    except OSError:
        pass
    
    return None


def create_model_directories(base_path: str) -> None:
    """
    Create all necessary model directories
    """
    model_dirs = [
        'checkpoints',
        'loras', 
        'vae',
        'embeddings',
        'controlnet',
        'clip_vision',
        'upscale_models'
    ]
    
    for model_dir in model_dirs:
        dir_path = os.path.join(base_path, model_dir)
        os.makedirs(dir_path, exist_ok=True) 