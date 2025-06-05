import json
import os
import hashlib
import base64
import requests
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import comfy.model_management
import folder_paths
import tempfile
import shutil
from typing import Dict, List, Optional, Tuple
import logging

from .config import (
    CIVITAI_SEARCH_ENDPOINT, 
    CIVITAI_MODEL_DETAIL_ENDPOINT,
    MODEL_TYPE_MAPPING,
    MODEL_EXTENSIONS,
    WORKFLOW_METADATA_FIELDS,
    MAX_SEARCH_RESULTS,
    DEFAULT_DOWNLOAD_CHUNK_SIZE,
    PROGRESS_LOG_INTERVAL
)
from .utils import (
    sanitize_filename,
    find_models_in_workflow,
    fuzzy_match_model_name,
    validate_workflow_json,
    find_existing_model_file,
    create_model_directories,
    format_file_size
)
from .settings import get_civitai_api_key, save_civitai_api_key

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CivitaiWorkflowParser:
    """
    ComfyUI Node for parsing Civitai PNG workflow images and automatically downloading missing models
    """
    
    def __init__(self):
        self.civitai_api_key = None
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "auto_download": ("BOOLEAN", {"default": False}),
                "check_only": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image": ("IMAGE",),  # Drag & drop image support
                "image_path": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Or enter path to PNG workflow image"
                }),
                "civitai_api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "API key (leave empty to use saved key)"
                }),
                "comfyui_models_path": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Custom ComfyUI models path (auto-detect if empty)"
                }),
                "prefer_safetensors": ("BOOLEAN", {"default": True}),
                "create_backup": ("BOOLEAN", {"default": False})
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("workflow_json", "missing_models", "download_status", "model_info")
    FUNCTION = "process_civitai_image"
    CATEGORY = "Civitai Helper"
    
    def process_civitai_image(self, auto_download: bool = False, check_only: bool = True,
                            image=None, image_path: str = "", civitai_api_key: str = "",
                            comfyui_models_path: str = "", prefer_safetensors: bool = True, 
                            create_backup: bool = False):
        """
        Main function to process Civitai PNG images and handle model downloads
        """
        # Get API key from settings if not provided
        if not civitai_api_key.strip():
            civitai_api_key = get_civitai_api_key()
        
        self.civitai_api_key = civitai_api_key.strip()
        self.prefer_safetensors = prefer_safetensors
        
        try:
            # Handle image input (drag & drop or path)
            target_image_path = None
            
            if image is not None:
                # Save the dragged image to a temporary file
                target_image_path = self.save_temp_image(image)
            elif image_path and os.path.exists(image_path):
                target_image_path = image_path
            else:
                return ("", "Error: Please drag & drop an image or provide a valid image path", "Failed", "")
            
            if not target_image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                return ("", "Error: File must be a PNG, JPG, or JPEG image", "Failed", "")
            
            # Extract workflow from PNG
            workflow_json = self.extract_workflow_from_png(target_image_path)
            if not workflow_json:
                return ("", "Failed to extract workflow from image. Make sure it contains ComfyUI metadata.", "Error", "")
            
            # Parse workflow and identify models
            missing_models = self.identify_missing_models(workflow_json, comfyui_models_path)
            
            # Generate model info summary
            model_info = self.generate_model_info_summary(workflow_json, missing_models)
            
            if check_only:
                status = "✅ Check completed - no downloads performed"
                if missing_models:
                    status += f"\n🔍 Found {len(missing_models)} missing models"
                    status += "\n💡 Set check_only=False and auto_download=True to download missing models"
                return (json.dumps(workflow_json, indent=2), 
                       json.dumps(missing_models, indent=2), 
                       status,
                       model_info)
            
            # Download missing models if auto_download is enabled
            download_status = "No missing models found"
            if missing_models and auto_download:
                if not self.civitai_api_key:
                    download_status = "❌ Error: Civitai API key is required for downloading models\n"
                    download_status += "💡 Get your API key from: https://civitai.com/user/account"
                else:
                    download_status = self.download_missing_models(missing_models, comfyui_models_path, create_backup)
            elif missing_models:
                download_status = f"🔍 Found {len(missing_models)} missing models\n"
                download_status += "💡 Set auto_download=True to download them automatically"
            
            return (json.dumps(workflow_json, indent=2), 
                   json.dumps(missing_models, indent=2), 
                   download_status,
                   model_info)
                   
        except Exception as e:
            logger.error(f"Error processing Civitai image: {str(e)}")
            return ("", f"Error: {str(e)}", "Failed", "")
        finally:
            # Clean up temporary file if it was created
            if image is not None and target_image_path and os.path.exists(target_image_path):
                try:
                    os.remove(target_image_path)
                except:
                    pass
    
    def save_temp_image(self, image):
        """
        Save a dragged image to a temporary file for processing
        """
        try:
            # Create a temporary file
            import tempfile
            import numpy as np
            from PIL import Image as PILImage
            
            # Convert tensor to PIL Image
            if hasattr(image, 'numpy'):
                image_array = image.numpy()
            else:
                image_array = image
            
            # Handle different tensor formats
            if len(image_array.shape) == 4:  # Batch dimension
                image_array = image_array[0]
            
            # Convert to 0-255 range if needed
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            else:
                image_array = image_array.astype(np.uint8)
            
            # Create PIL Image
            pil_image = PILImage.fromarray(image_array)
            
            # Save to temporary file
            temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
            os.close(temp_fd)
            pil_image.save(temp_path, 'PNG')
            
            return temp_path
            
        except Exception as e:
            logger.error(f"Error saving temporary image: {str(e)}")
            return None
    
    def extract_workflow_from_png(self, image_path: str) -> Optional[Dict]:
        """
        Extract ComfyUI workflow JSON from PNG metadata with enhanced detection
        """
        try:
            with Image.open(image_path) as img:
                # Check for workflow in text chunks
                if hasattr(img, 'text'):
                    for key in WORKFLOW_METADATA_FIELDS:
                        if key in img.text:
                            workflow_data = img.text[key]
                            workflow = validate_workflow_json(workflow_data)
                            if workflow:
                                logger.info(f"Found workflow in text metadata: {key}")
                                return workflow
                
                # Check PNG info chunks
                if hasattr(img, 'info'):
                    for key, value in img.info.items():
                        if any(field.lower() in key.lower() for field in WORKFLOW_METADATA_FIELDS):
                            workflow = validate_workflow_json(str(value))
                            if workflow:
                                logger.info(f"Found workflow in PNG info: {key}")
                                return workflow
                
                # Try to find base64 encoded workflow data
                if hasattr(img, 'text'):
                    for key, value in img.text.items():
                        if len(value) > 100 and self.looks_like_base64(value):
                            try:
                                decoded = base64.b64decode(value).decode('utf-8')
                                workflow = validate_workflow_json(decoded)
                                if workflow:
                                    logger.info(f"Found base64 encoded workflow in: {key}")
                                    return workflow
                            except Exception:
                                continue
                
                logger.warning("No valid workflow metadata found in image")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting workflow from image: {str(e)}")
            return None
    
    def looks_like_base64(self, data: str) -> bool:
        """Check if string looks like base64 encoded data"""
        try:
            if len(data) % 4 != 0:
                return False
            base64.b64decode(data, validate=True)
            return True
        except Exception:
            return False
    
    def identify_missing_models(self, workflow: Dict, custom_models_path: str = "") -> List[Dict]:
        """
        Identify missing models from the workflow using enhanced detection
        """
        # Get ComfyUI models path
        if custom_models_path and os.path.exists(custom_models_path):
            models_base_path = custom_models_path
        else:
            models_base_path = folder_paths.models_dir
        
        # Ensure model directories exist
        create_model_directories(models_base_path)
        
        # Find all models referenced in workflow
        referenced_models = find_models_in_workflow(workflow)
        
        missing_models = []
        for model in referenced_models:
            model_name = model['name']
            model_type = model['type']
            
            # Check if model exists
            model_dir = os.path.join(models_base_path, model_type)
            existing_file = find_existing_model_file(model_name, model_dir, model_type)
            
            if not existing_file:
                missing_models.append({
                    'name': model_name,
                    'type': model_type,
                    'node_id': model['node_id'],
                    'class_type': model.get('class_type', ''),
                    'input_key': model.get('input_key', ''),
                    'search_attempted': False,
                    'civitai_id': None,
                    'download_url': None
                })
            else:
                logger.info(f"Model found: {model_name} at {existing_file}")
        
        return missing_models
    
    def generate_model_info_summary(self, workflow: Dict, missing_models: List[Dict]) -> str:
        """
        Generate a summary of all models in the workflow
        """
        all_models = find_models_in_workflow(workflow)
        
        summary = f"Workflow Model Summary:\n"
        summary += f"Total models referenced: {len(all_models)}\n"
        summary += f"Missing models: {len(missing_models)}\n\n"
        
        # Group by type
        by_type = {}
        for model in all_models:
            model_type = model['type']
            if model_type not in by_type:
                by_type[model_type] = []
            by_type[model_type].append(model['name'])
        
        for model_type, models in by_type.items():
            summary += f"{model_type.title()}: {len(models)} models\n"
            for model_name in models:
                is_missing = any(m['name'] == model_name for m in missing_models)
                status = "❌ MISSING" if is_missing else "✅ Found"
                summary += f"  • {model_name} [{status}]\n"
            summary += "\n"
        
        return summary
    
    def download_missing_models(self, missing_models: List[Dict], custom_models_path: str = "", 
                              create_backup: bool = False) -> str:
        """
        Download missing models using Civitai API with enhanced error handling
        """
        if not self.civitai_api_key:
            return "Error: Civitai API key is required for downloading models"
        
        # Get ComfyUI models path
        if custom_models_path and os.path.exists(custom_models_path):
            models_base_path = custom_models_path
        else:
            models_base_path = folder_paths.models_dir
        
        download_results = []
        successful_downloads = 0
        
        for i, model in enumerate(missing_models):
            try:
                logger.info(f"Processing model {i+1}/{len(missing_models)}: {model['name']}")
                
                # Search for model on Civitai
                model_info = self.search_civitai_model(model['name'])
                if model_info:
                    download_result = self.download_model_from_civitai(
                        model_info, model['type'], models_base_path, create_backup
                    )
                    if "Successfully downloaded" in download_result:
                        successful_downloads += 1
                    download_results.append(f"{model['name']}: {download_result}")
                else:
                    download_results.append(f"{model['name']}: Not found on Civitai")
            
            except Exception as e:
                error_msg = f"Error downloading {model['name']}: {str(e)}"
                logger.error(error_msg)
                download_results.append(error_msg)
        
        # Summary
        summary = f"📥 Download Summary: {successful_downloads}/{len(missing_models)} models downloaded successfully\n\n"
        summary += "\n".join(download_results)
        
        return summary
    
    def search_civitai_model(self, model_name: str) -> Optional[Dict]:
        """
        Search for a model on Civitai using the API with improved matching
        """
        try:
            # Clean model name for search
            search_name = os.path.splitext(model_name)[0]
            
            headers = {
                'Authorization': f'Bearer {self.civitai_api_key}',
                'Content-Type': 'application/json'
            }
            
            # Try different search strategies
            search_terms = [
                search_name,  # Original name
                search_name.replace('_', ' '),  # Replace underscores with spaces
                search_name.replace('-', ' '),  # Replace hyphens with spaces
            ]
            
            for search_term in search_terms:
                params = {
                    'query': search_term,
                    'limit': MAX_SEARCH_RESULTS
                }
                
                response = requests.get(CIVITAI_SEARCH_ENDPOINT, headers=headers, params=params, timeout=30)
                response.raise_for_status()
                
                search_results = response.json()
                
                if 'items' in search_results and search_results['items']:
                    # Find the best match
                    for item in search_results['items']:
                        if fuzzy_match_model_name(search_name, item['name']):
                            # Get detailed model info including files
                            detail_response = requests.get(
                                f"{CIVITAI_MODEL_DETAIL_ENDPOINT}/{item['id']}", 
                                headers=headers,
                                timeout=30
                            )
                            detail_response.raise_for_status()
                            return detail_response.json()
            
            return None
            
        except requests.RequestException as e:
            logger.error(f"Network error searching Civitai: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error searching Civitai: {str(e)}")
            return None
    
    def download_model_from_civitai(self, model_info: Dict, model_type: str, models_base_path: str,
                                  create_backup: bool = False) -> str:
        """
        Download a specific model file from Civitai with enhanced file selection
        """
        try:
            # Find the best file to download
            if 'modelVersions' not in model_info or not model_info['modelVersions']:
                return "No versions available"
            
            latest_version = model_info['modelVersions'][0]
            if 'files' not in latest_version or not latest_version['files']:
                return "No files available"
            
            # Select best file based on preferences
            best_file = self.select_best_file(latest_version['files'])
            if not best_file:
                return "No suitable files found"
            
            # Create target directory
            target_dir = os.path.join(models_base_path, model_type)
            os.makedirs(target_dir, exist_ok=True)
            
            # Sanitize filename
            file_name = sanitize_filename(best_file['name'])
            target_path = os.path.join(target_dir, file_name)
            
            # Check if file already exists
            if os.path.exists(target_path):
                if create_backup:
                    backup_path = f"{target_path}.backup"
                    shutil.move(target_path, backup_path)
                    logger.info(f"Created backup: {backup_path}")
                else:
                    return f"File already exists: {target_path}"
            
            # Download file
            download_url = best_file['downloadUrl']
            
            headers = {
                'Authorization': f'Bearer {self.civitai_api_key}'
            }
            
            logger.info(f"Starting download: {file_name}")
            response = requests.get(download_url, headers=headers, stream=True, timeout=60)
            response.raise_for_status()
            
            # Get file size
            total_size = int(response.headers.get('content-length', 0))
            size_str = format_file_size(total_size) if total_size > 0 else "Unknown size"
            
            # Download with progress tracking
            with open(target_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=DEFAULT_DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Log progress every 10MB
                        if downloaded % PROGRESS_LOG_INTERVAL == 0 and total_size > 0:
                            progress = (downloaded / total_size) * 100
                            logger.info(f"Downloading {file_name}: {progress:.1f}% ({format_file_size(downloaded)}/{size_str})")
            
            # Verify download
            if total_size > 0 and downloaded != total_size:
                os.remove(target_path)
                return f"Download incomplete. Expected {size_str}, got {format_file_size(downloaded)}"
            
            logger.info(f"Download completed: {file_name} ({format_file_size(downloaded)})")
            return f"Successfully downloaded to {target_path} ({format_file_size(downloaded)})"
            
        except requests.RequestException as e:
            logger.error(f"Network error downloading model: {str(e)}")
            return f"Network error: {str(e)}"
        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            return f"Download failed: {str(e)}"
    
    def select_best_file(self, files: List[Dict]) -> Optional[Dict]:
        """
        Select the best file to download based on preferences
        """
        if not files:
            return None
        
        # Prioritize safetensors if preferred
        if self.prefer_safetensors:
            safetensors_files = [f for f in files if f['name'].endswith('.safetensors')]
            if safetensors_files:
                return safetensors_files[0]
        
        # Look for other supported formats
        supported_extensions = ['.safetensors', '.pt', '.ckpt', '.bin', '.pth']
        for ext in supported_extensions:
            matching_files = [f for f in files if f['name'].endswith(ext)]
            if matching_files:
                return matching_files[0]
        
        # Fall back to first file
        return files[0]


class CivitaiModelDownloader:
    """
    Dedicated node for downloading missing models from a workflow
    """
    
    def __init__(self):
        self.civitai_api_key = None
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow_json": ("STRING", {"forceInput": True}),
                "missing_models": ("STRING", {"forceInput": True}),
                "download_models": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "civitai_api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "API key (leave empty to use saved key)"
                }),
                "comfyui_models_path": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Custom ComfyUI models path (auto-detect if empty)"
                }),
                "prefer_safetensors": ("BOOLEAN", {"default": True}),
                "create_backup": ("BOOLEAN", {"default": False})
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("download_status", "download_summary")
    FUNCTION = "download_models"
    CATEGORY = "Civitai Helper"
    
    def download_models(self, workflow_json: str, missing_models: str, download_models: bool = False,
                       civitai_api_key: str = "", comfyui_models_path: str = "",
                       prefer_safetensors: bool = True, create_backup: bool = False):
        """
        Download missing models from Civitai
        """
        if not download_models:
            return ("⏸️ Download disabled. Set download_models=True to proceed.", 
                   "💡 This is a safety feature to prevent accidental downloads.")
        
        # Get API key from settings if not provided
        if not civitai_api_key.strip():
            civitai_api_key = get_civitai_api_key()
        
        if not civitai_api_key:
            return ("❌ Error: Civitai API key is required for downloading models", 
                   "💡 Get your API key from: https://civitai.com/user/account")
        
        try:
            # Parse missing models
            missing_models_list = json.loads(missing_models)
            if not missing_models_list:
                return ("✅ No missing models to download", "All models are already available")
            
            # Initialize downloader
            parser = CivitaiWorkflowParser()
            parser.civitai_api_key = civitai_api_key
            parser.prefer_safetensors = prefer_safetensors
            
            # Download models
            download_result = parser.download_missing_models(
                missing_models_list, comfyui_models_path, create_backup
            )
            
            # Generate summary
            successful = download_result.count("Successfully downloaded")
            total = len(missing_models_list)
            
            summary = f"🎯 Download Complete!\n"
            summary += f"✅ Successfully downloaded: {successful}/{total} models\n"
            if successful < total:
                summary += f"❌ Failed downloads: {total - successful}\n"
            summary += f"\n📋 Details:\n{download_result}"
            
            return (f"Downloaded {successful}/{total} models successfully", summary)
            
        except json.JSONDecodeError:
            return ("❌ Error: Invalid missing models data", "Failed to parse missing models list")
        except Exception as e:
            return (f"❌ Error: {str(e)}", f"Download failed: {str(e)}")


class CivitaiSettingsNode:
    """
    Node for managing Civitai API key settings
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["Get Current Key", "Save New Key", "Clear Saved Key"], {"default": "Get Current Key"}),
            },
            "optional": {
                "new_api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Enter your Civitai API key to save"
                })
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("status", "current_key_info")
    FUNCTION = "manage_settings"
    CATEGORY = "Civitai Helper"
    
    def manage_settings(self, action: str, new_api_key: str = ""):
        """
        Manage Civitai API key settings
        """
        try:
            if action == "Get Current Key":
                current_key = get_civitai_api_key()
                if current_key:
                    masked_key = f"{current_key[:8]}..." if len(current_key) > 8 else "***"
                    return ("✅ API key is saved", f"Current key: {masked_key}")
                else:
                    return ("ℹ️ No API key saved", "No API key found in settings")
            
            elif action == "Save New Key":
                if not new_api_key.strip():
                    return ("❌ Error: Please provide an API key to save", "")
                
                save_civitai_api_key(new_api_key.strip())
                masked_key = f"{new_api_key[:8]}..." if len(new_api_key) > 8 else "***"
                return ("✅ API key saved successfully", f"Saved key: {masked_key}")
            
            elif action == "Clear Saved Key":
                save_civitai_api_key("")
                return ("✅ API key cleared", "API key has been removed from settings")
            
        except Exception as e:
            return (f"❌ Error: {str(e)}", "Failed to manage settings") 