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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CivitaiHelper:
    """
    Comprehensive ComfyUI Node for Civitai workflow image processing and model downloading
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upload_image": ("UPLOAD",),
            },
            "optional": {
                "civitai_api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Enter your Civitai API key"
                }),
                "comfyui_models_path": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Custom models path (leave empty for auto-detect)"
                }),
                "auto_download": ("BOOLEAN", {"default": False}),
                "prefer_safetensors": ("BOOLEAN", {"default": True}),
                "create_backup": ("BOOLEAN", {"default": False}),
                "progress_log": ("STRING", {
                    "multiline": True,
                    "default": "Ready to process Civitai workflow image...\n\n1. Click 'Upload' to select a PNG image with ComfyUI workflow metadata\n2. Enter your Civitai API key if you want to download missing models\n3. Optionally set a custom models path\n4. Enable auto_download to automatically download missing models\n5. Execute the node to analyze the workflow",
                    "placeholder": "Progress and results will appear here..."
                }),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "process_workflow_image"
    CATEGORY = "Civitai Helper"
    OUTPUT_NODE = True
    
    def process_workflow_image(self, upload_image=None, civitai_api_key: str = "", 
                             comfyui_models_path: str = "", auto_download: bool = False,
                             prefer_safetensors: bool = True, create_backup: bool = False,
                             progress_log: str = ""):
        """
        Main function to process Civitai workflow images and handle model downloads
        """
        # Initialize progress log
        log_lines = ["🚀 Starting Civitai Workflow Analysis", "=" * 50]
        
        try:
            # Handle image upload
            if not upload_image:
                log_lines.append("❌ No image uploaded. Please upload a PNG image with ComfyUI workflow metadata.")
                return {"ui": {"text": ["\n".join(log_lines)]}}
            
            # Get image path from upload
            image_path = self.handle_uploaded_image(upload_image, log_lines)
            if not image_path:
                return {"ui": {"text": ["\n".join(log_lines)]}}
            
            # Extract workflow from image
            log_lines.append("\n🔍 Extracting workflow metadata from image...")
            workflow_json = self.extract_workflow_from_image(image_path, log_lines)
            
            if not workflow_json:
                log_lines.append("❌ No ComfyUI workflow found in image metadata")
                log_lines.append("💡 Make sure the image was exported from ComfyUI with workflow metadata")
                return {"ui": {"text": ["\n".join(log_lines)]}}
            
            log_lines.append("✅ Successfully extracted workflow from image")
            log_lines.append(f"📊 Workflow contains {len(workflow_json.get('nodes', []))} nodes")
            
            # Get models path
            models_path = comfyui_models_path.strip() if comfyui_models_path.strip() else folder_paths.models_dir
            log_lines.append(f"📁 Using models directory: {models_path}")
            
            # Analyze workflow for models
            log_lines.append("\n🔍 Analyzing workflow for model dependencies...")
            missing_models = self.analyze_workflow_models(workflow_json, models_path, log_lines)
            
            # Display results
            self.display_model_analysis_results(workflow_json, missing_models, log_lines)
            
            # Handle downloads if requested
            if auto_download and missing_models:
                if not civitai_api_key.strip():
                    log_lines.append("\n❌ Auto-download enabled but no API key provided")
                    log_lines.append("💡 Get your API key from: https://civitai.com/user/account")
                else:
                    log_lines.append(f"\n📥 Starting automatic download of {len(missing_models)} missing models...")
                    self.download_missing_models(missing_models, civitai_api_key.strip(), models_path, 
                                               prefer_safetensors, create_backup, log_lines)
            elif missing_models and not auto_download:
                log_lines.append("\n💡 Enable 'auto_download' to automatically download missing models")
            
            log_lines.append("\n✨ Analysis complete!")
            
        except Exception as e:
            log_lines.append(f"\n💥 Unexpected error: {str(e)}")
            logger.error(f"Error in CivitaiHelper: {str(e)}", exc_info=True)
        
        return {"ui": {"text": ["\n".join(log_lines)]}}
    
    def handle_uploaded_image(self, upload_image, log_lines: List[str]) -> Optional[str]:
        """
        Handle the uploaded image and return its path
        """
        try:
            # Handle different upload formats
            if hasattr(upload_image, 'name'):
                image_path = upload_image.name
                log_lines.append(f"📁 Uploaded image: {os.path.basename(image_path)}")
            elif isinstance(upload_image, str):
                image_path = upload_image
                log_lines.append(f"📁 Image path: {os.path.basename(image_path)}")
            else:
                log_lines.append(f"❓ Unknown upload format: {type(upload_image)}")
                return None
            
            # Validate file exists
            if not os.path.exists(image_path):
                log_lines.append(f"❌ Image file not found: {image_path}")
                return None
            
            # Validate file type
            if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                log_lines.append(f"❌ Unsupported file type. Must be PNG, JPG, or JPEG")
                return None
            
            # Get file info
            file_size = os.path.getsize(image_path)
            log_lines.append(f"📏 File size: {format_file_size(file_size)}")
            
            return image_path
            
        except Exception as e:
            log_lines.append(f"❌ Error handling uploaded image: {str(e)}")
            return None
    
    def extract_workflow_from_image(self, image_path: str, log_lines: List[str]) -> Optional[Dict]:
        """
        Extract workflow from image with detailed logging
        """
        try:
            with Image.open(image_path) as img:
                log_lines.append(f"🖼️ Image: {img.format} {img.size} {img.mode}")
                
                # Check text metadata
                if hasattr(img, 'text'):
                    log_lines.append(f"📝 Found {len(img.text)} text metadata fields")
                    
                    for field in WORKFLOW_METADATA_FIELDS:
                        if field in img.text:
                            log_lines.append(f"🔍 Found workflow in field: {field}")
                            workflow_data = img.text[field]
                            
                            workflow = validate_workflow_json(workflow_data)
                            if workflow:
                                log_lines.append(f"✅ Successfully parsed workflow JSON ({len(workflow_data)} chars)")
                                return workflow
                            else:
                                log_lines.append(f"❌ Invalid JSON in field: {field}")
                
                # Check PNG info
                if hasattr(img, 'info'):
                    log_lines.append(f"📝 Found {len(img.info)} PNG info fields")
                    
                    for key, value in img.info.items():
                        if any(field.lower() in key.lower() for field in WORKFLOW_METADATA_FIELDS):
                            log_lines.append(f"🔍 Found workflow in PNG info: {key}")
                            workflow = validate_workflow_json(str(value))
                            if workflow:
                                log_lines.append(f"✅ Successfully parsed workflow from PNG info")
                                return workflow
                
                # List available metadata for debugging
                if hasattr(img, 'text'):
                    log_lines.append("📋 Available text metadata fields:")
                    for key in list(img.text.keys())[:10]:  # Limit to first 10
                        log_lines.append(f"   • {key}")
                
                return None
                
        except Exception as e:
            log_lines.append(f"💥 Error reading image: {str(e)}")
            return None
    
    def analyze_workflow_models(self, workflow: Dict, models_path: str, log_lines: List[str]) -> List[Dict]:
        """
        Analyze workflow for model dependencies
        """
        try:
            # Ensure model directories exist
            create_model_directories(models_path)
            
            # Find all models in workflow
            referenced_models = find_models_in_workflow(workflow)
            log_lines.append(f"🔍 Found {len(referenced_models)} model references in workflow")
            
            missing_models = []
            found_models = []
            
            for model in referenced_models:
                model_name = model['name']
                model_type = model['type']
                
                # Check if model exists
                model_dir = os.path.join(models_path, model_type)
                existing_file = find_existing_model_file(model_name, model_dir, model_type)
                
                if existing_file:
                    found_models.append(model)
                    log_lines.append(f"✅ {model_type}: {model_name}")
                else:
                    missing_models.append(model)
                    log_lines.append(f"❌ {model_type}: {model_name}")
            
            log_lines.append(f"\n📊 Model Analysis Summary:")
            log_lines.append(f"   • Total models: {len(referenced_models)}")
            log_lines.append(f"   • Found: {len(found_models)}")
            log_lines.append(f"   • Missing: {len(missing_models)}")
            
            return missing_models
            
        except Exception as e:
            log_lines.append(f"💥 Error analyzing models: {str(e)}")
            return []
    
    def display_model_analysis_results(self, workflow: Dict, missing_models: List[Dict], log_lines: List[str]):
        """
        Display detailed analysis results
        """
        all_models = find_models_in_workflow(workflow)
        
        # Group by type
        by_type = {}
        for model in all_models:
            model_type = model['type']
            if model_type not in by_type:
                by_type[model_type] = {'total': 0, 'missing': 0, 'found': 0}
            by_type[model_type]['total'] += 1
            
            if any(m['name'] == model['name'] for m in missing_models):
                by_type[model_type]['missing'] += 1
            else:
                by_type[model_type]['found'] += 1
        
        if by_type:
            log_lines.append(f"\n📋 Detailed Model Breakdown:")
            for model_type, stats in by_type.items():
                status_icon = "✅" if stats['missing'] == 0 else "⚠️"
                log_lines.append(f"   {status_icon} {model_type.title()}: {stats['found']}/{stats['total']} found")
        
        if missing_models:
            log_lines.append(f"\n❌ Missing Models ({len(missing_models)}):")
            for model in missing_models:
                log_lines.append(f"   • {model['name']} ({model['type']})")
    
    def download_missing_models(self, missing_models: List[Dict], api_key: str, models_path: str,
                              prefer_safetensors: bool, create_backup: bool, log_lines: List[str]):
        """
        Download missing models from Civitai
        """
        successful_downloads = 0
        
        for i, model in enumerate(missing_models, 1):
            try:
                log_lines.append(f"\n📥 [{i}/{len(missing_models)}] Searching for: {model['name']}")
                
                # Search for model on Civitai
                model_info = self.search_civitai_model(model['name'], api_key, log_lines)
                
                if model_info:
                    # Download the model
                    download_result = self.download_model_from_civitai(
                        model_info, model['type'], models_path, api_key, 
                        prefer_safetensors, create_backup, log_lines
                    )
                    
                    if "Successfully downloaded" in download_result:
                        successful_downloads += 1
                        log_lines.append(f"✅ {download_result}")
                    else:
                        log_lines.append(f"❌ {download_result}")
                else:
                    log_lines.append(f"❌ Model not found on Civitai: {model['name']}")
                    
            except Exception as e:
                log_lines.append(f"💥 Error downloading {model['name']}: {str(e)}")
        
        # Summary
        log_lines.append(f"\n🎯 Download Summary:")
        log_lines.append(f"   • Successfully downloaded: {successful_downloads}/{len(missing_models)}")
        log_lines.append(f"   • Failed downloads: {len(missing_models) - successful_downloads}")
        
        if successful_downloads > 0:
            log_lines.append(f"\n✨ Downloaded models are now available in ComfyUI!")
    
    def search_civitai_model(self, model_name: str, api_key: str, log_lines: List[str]) -> Optional[Dict]:
        """
        Search for model on Civitai
        """
        try:
            search_name = os.path.splitext(model_name)[0]
            
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            params = {
                'query': search_name,
                'limit': MAX_SEARCH_RESULTS
            }
            
            response = requests.get(CIVITAI_SEARCH_ENDPOINT, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            
            search_results = response.json()
            
            if 'items' in search_results and search_results['items']:
                for item in search_results['items']:
                    if fuzzy_match_model_name(search_name, item['name']):
                        log_lines.append(f"🎯 Found match: {item['name']}")
                        
                        # Get detailed model info
                        detail_response = requests.get(
                            f"{CIVITAI_MODEL_DETAIL_ENDPOINT}/{item['id']}", 
                            headers=headers, timeout=30
                        )
                        detail_response.raise_for_status()
                        return detail_response.json()
            
            return None
            
        except Exception as e:
            log_lines.append(f"💥 Search error: {str(e)}")
            return None
    
    def download_model_from_civitai(self, model_info: Dict, model_type: str, models_path: str,
                                  api_key: str, prefer_safetensors: bool, create_backup: bool,
                                  log_lines: List[str]) -> str:
        """
        Download model file from Civitai
        """
        try:
            if 'modelVersions' not in model_info or not model_info['modelVersions']:
                return "No versions available"
            
            latest_version = model_info['modelVersions'][0]
            if 'files' not in latest_version or not latest_version['files']:
                return "No files available"
            
            # Select best file
            best_file = self.select_best_file(latest_version['files'], prefer_safetensors)
            if not best_file:
                return "No suitable files found"
            
            # Prepare download
            target_dir = os.path.join(models_path, model_type)
            os.makedirs(target_dir, exist_ok=True)
            
            file_name = sanitize_filename(best_file['name'])
            target_path = os.path.join(target_dir, file_name)
            
            # Check if exists
            if os.path.exists(target_path):
                if create_backup:
                    backup_path = f"{target_path}.backup"
                    shutil.move(target_path, backup_path)
                    log_lines.append(f"📦 Created backup: {os.path.basename(backup_path)}")
                else:
                    return f"File already exists: {file_name}"
            
            # Download
            download_url = best_file['downloadUrl']
            headers = {'Authorization': f'Bearer {api_key}'}
            
            log_lines.append(f"⬇️ Downloading: {file_name}")
            response = requests.get(download_url, headers=headers, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(target_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=DEFAULT_DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
            
            return f"Successfully downloaded {file_name} ({format_file_size(downloaded)})"
            
        except Exception as e:
            return f"Download failed: {str(e)}"
    
    def select_best_file(self, files: List[Dict], prefer_safetensors: bool) -> Optional[Dict]:
        """
        Select the best file to download
        """
        if not files:
            return None
        
        if prefer_safetensors:
            safetensors_files = [f for f in files if f['name'].endswith('.safetensors')]
            if safetensors_files:
                return safetensors_files[0]
        
        # Look for other supported formats
        supported_extensions = ['.safetensors', '.pt', '.ckpt', '.bin', '.pth']
        for ext in supported_extensions:
            matching_files = [f for f in files if f['name'].endswith(ext)]
            if matching_files:
                return matching_files[0]
        
        return files[0] 