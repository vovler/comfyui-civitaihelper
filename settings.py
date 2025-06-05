"""
Settings management for Civitai Helper ComfyUI Extension
"""

import os
import json
import folder_paths
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Settings file path
SETTINGS_FILE = os.path.join(folder_paths.base_path, "custom_nodes", "civitai_helper_settings.json")

def get_settings_file_path() -> str:
    """
    Get the path to the settings file
    """
    # Try to place in custom_nodes directory first
    custom_nodes_path = os.path.join(folder_paths.base_path, "custom_nodes", "civitai_helper_settings.json")
    
    # If that doesn't work, try user directory
    if not os.path.exists(os.path.dirname(custom_nodes_path)):
        try:
            user_dir = os.path.expanduser("~")
            custom_nodes_path = os.path.join(user_dir, ".comfyui_civitai_settings.json")
        except:
            # Fall back to current directory
            custom_nodes_path = "civitai_helper_settings.json"
    
    return custom_nodes_path

def load_settings() -> dict:
    """
    Load settings from file
    """
    settings_path = get_settings_file_path()
    
    try:
        if os.path.exists(settings_path):
            with open(settings_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load settings: {str(e)}")
    
    return {}

def save_settings(settings: dict) -> bool:
    """
    Save settings to file
    """
    settings_path = get_settings_file_path()
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(settings_path), exist_ok=True)
        
        with open(settings_path, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2)
        
        logger.info(f"Settings saved to: {settings_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save settings: {str(e)}")
        return False

def get_civitai_api_key() -> Optional[str]:
    """
    Get the saved Civitai API key
    """
    # First, try environment variable
    api_key = os.environ.get('CIVITAI_API_KEY')
    if api_key:
        return api_key.strip()
    
    # Then try settings file
    settings = load_settings()
    api_key = settings.get('civitai_api_key', '')
    
    return api_key.strip() if api_key else None

def save_civitai_api_key(api_key: str) -> bool:
    """
    Save the Civitai API key to settings
    """
    settings = load_settings()
    settings['civitai_api_key'] = api_key.strip()
    
    return save_settings(settings)

def get_models_path() -> Optional[str]:
    """
    Get the saved ComfyUI models path
    """
    # First, try environment variable
    models_path = os.environ.get('COMFYUI_MODELS_PATH')
    if models_path and os.path.exists(models_path):
        return models_path
    
    # Then try settings file
    settings = load_settings()
    models_path = settings.get('comfyui_models_path', '')
    
    if models_path and os.path.exists(models_path):
        return models_path
    
    # Fall back to ComfyUI's default
    try:
        return folder_paths.models_dir
    except:
        return None

def save_models_path(models_path: str) -> bool:
    """
    Save the ComfyUI models path to settings
    """
    settings = load_settings()
    settings['comfyui_models_path'] = models_path.strip()
    
    return save_settings(settings)

def get_user_preferences() -> dict:
    """
    Get user preferences
    """
    settings = load_settings()
    
    return {
        'prefer_safetensors': settings.get('prefer_safetensors', True),
        'create_backup': settings.get('create_backup', False),
        'auto_download': settings.get('auto_download', False),
        'max_concurrent_downloads': settings.get('max_concurrent_downloads', 3),
        'download_timeout': settings.get('download_timeout', 300),
    }

def save_user_preferences(preferences: dict) -> bool:
    """
    Save user preferences
    """
    settings = load_settings()
    
    # Update preferences
    for key, value in preferences.items():
        if key in ['prefer_safetensors', 'create_backup', 'auto_download', 
                  'max_concurrent_downloads', 'download_timeout']:
            settings[key] = value
    
    return save_settings(settings)

def clear_all_settings() -> bool:
    """
    Clear all saved settings
    """
    settings_path = get_settings_file_path()
    
    try:
        if os.path.exists(settings_path):
            os.remove(settings_path)
            logger.info("All settings cleared")
        return True
    except Exception as e:
        logger.error(f"Failed to clear settings: {str(e)}")
        return False

def export_settings() -> str:
    """
    Export settings as JSON string (without sensitive data)
    """
    settings = load_settings()
    
    # Remove sensitive data for export
    export_settings = settings.copy()
    if 'civitai_api_key' in export_settings:
        # Mask the API key
        api_key = export_settings['civitai_api_key']
        if len(api_key) > 8:
            export_settings['civitai_api_key'] = f"{api_key[:4]}...{api_key[-4:]}"
        else:
            export_settings['civitai_api_key'] = "***"
    
    return json.dumps(export_settings, indent=2)

def import_settings(settings_json: str, include_api_key: bool = False) -> bool:
    """
    Import settings from JSON string
    """
    try:
        new_settings = json.loads(settings_json)
        current_settings = load_settings()
        
        # Update settings, optionally excluding API key
        for key, value in new_settings.items():
            if key == 'civitai_api_key' and not include_api_key:
                continue  # Skip API key import for security
            current_settings[key] = value
        
        return save_settings(current_settings)
        
    except Exception as e:
        logger.error(f"Failed to import settings: {str(e)}")
        return False

def validate_api_key(api_key: str) -> bool:
    """
    Basic validation of API key format
    """
    if not api_key or not isinstance(api_key, str):
        return False
    
    # Basic checks - Civitai API keys are typically long alphanumeric strings
    api_key = api_key.strip()
    
    if len(api_key) < 10:
        return False
    
    # Check if it contains only valid characters (alphanumeric and some special chars)
    import re
    if not re.match(r'^[a-zA-Z0-9._-]+$', api_key):
        return False
    
    return True

def get_settings_info() -> dict:
    """
    Get information about current settings
    """
    settings_path = get_settings_file_path()
    info = {
        'settings_file_path': settings_path,
        'settings_file_exists': os.path.exists(settings_path),
        'has_api_key': bool(get_civitai_api_key()),
        'models_path': get_models_path(),
        'preferences': get_user_preferences()
    }
    
    if info['settings_file_exists']:
        try:
            stat = os.stat(settings_path)
            import time
            info['last_modified'] = time.ctime(stat.st_mtime)
            info['file_size'] = stat.st_size
        except:
            pass
    
    return info 