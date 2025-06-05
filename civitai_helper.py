import json
import os
import hashlib
import base64
import requests
from PIL import Image, ImageSequence, ImageOps
from PIL.PngImagePlugin import PngInfo
import comfy.model_management
import folder_paths
import tempfile
import shutil
import torch
import numpy as np
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
    def INPUT_TYPES(s):
        return {
            "required": {
                "load_test_image": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Load test.png from ComfyUI/input/ folder"
                }),
            },
            "optional": {
                "civitai_api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "🔑 Enter your Civitai API key",
                    "tooltip": "Get your API key from https://civitai.com/user/account"
                }),
                "comfyui_models_path": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "📂 Custom models path (leave empty for auto-detect)",
                    "tooltip": "Override the default ComfyUI models directory"
                }),
                "auto_download": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Automatically download missing models from Civitai"
                }),
                "prefer_safetensors": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Prefer .safetensors files over other formats"
                }),
                "create_backup": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Create backup of existing files before overwriting"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("🖼️ image_preview", "📋 progress_log")
    FUNCTION = "process_workflow_image"
    CATEGORY = "Civitai Helper"
    OUTPUT_NODE = True
    
    def process_workflow_image(self, load_test_image: bool = True, civitai_api_key: str = "", 
                             comfyui_models_path: str = "", auto_download: bool = False,
                             prefer_safetensors: bool = True, create_backup: bool = False):
        """
        Main function to process Civitai workflow images and handle model downloads
        """
        # Initialize progress log
        log_lines = [
            "🎯 Civitai Helper - ComfyUI Workflow Analyzer",
            "=" * 50,
            "",
            "📋 Test Mode - Loading from fixed path:",
            "🔍 Looking for: ComfyUI/input/test.png",
            "",
            "💡 Place your workflow PNG image at ComfyUI/input/test.png",
            "📝 The image should contain ComfyUI workflow metadata",
            "",
        ]
        
        # Default image tensor (black image)
        default_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
        
        try:
            # Build path to test.png
            input_dir = folder_paths.get_input_directory()
            image_path = os.path.join(input_dir, "test.png")
            
            log_lines.append(f"📁 Looking for image at: {image_path}")
            
            # Check if file exists
            if not os.path.exists(image_path):
                log_lines.extend([
                    "",
                    f"❌ Image file not found: {image_path}",
                    "",
                    "📝 To test this node:",
                    "   1. Place a PNG workflow image at ComfyUI/input/test.png",
                    "   2. Make sure it's a PNG exported from ComfyUI with workflow metadata",
                    "   3. Re-execute this node",
                    "",
                    f"🔍 Expected path: {image_path}",
                    f"📁 Input directory: {input_dir}"
                ])
                return (default_image, "\n".join(log_lines))
            
            log_lines.append(f"✅ Found image file: {os.path.basename(image_path)}")
            
            # Load image for preview
            image_tensor = self.load_image_for_preview(image_path, log_lines)
            
            # Extract workflow from the ORIGINAL image file (preserves metadata)
            log_lines.append("\n🔍 Extracting workflow metadata from image...")
            workflow_json = self.extract_workflow_from_image(image_path, log_lines)
            
            if not workflow_json:
                log_lines.extend([
                    "",
                    "❌ No ComfyUI workflow found in image metadata",
                    "",
                    "💡 Troubleshooting:",
                    "   • Make sure the image was exported from ComfyUI (not just saved)",
                    "   • Check that 'Save Workflow' is enabled in ComfyUI settings",
                    "   • Try using a PNG image (JPG may not preserve metadata)",
                    "   • The image should have been generated with ComfyUI, not just opened in it",
                    "",
                    "🔍 This tool analyzes the original image file for metadata",
                ])
                return (image_tensor, "\n".join(log_lines))
            
            log_lines.extend([
                "✅ Successfully extracted workflow from image!",
                f"📊 Workflow contains {len(workflow_json.get('nodes', []))} nodes",
                ""
            ])
            
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
                    log_lines.extend([
                        "",
                        "❌ Auto-download enabled but no API key provided",
                        "",
                        "🔑 To enable downloads:",
                        "   1. Go to https://civitai.com/user/account",
                        "   2. Copy your API key",
                        "   3. Paste it in the 'civitai_api_key' field above",
                        "   4. Re-execute this node"
                    ])
                else:
                    log_lines.append(f"\n📥 Starting automatic download of {len(missing_models)} missing models...")
                    self.download_missing_models(missing_models, civitai_api_key.strip(), models_path, 
                                               prefer_safetensors, create_backup, log_lines)
            elif missing_models and not auto_download:
                log_lines.extend([
                    "",
                    "💡 To download missing models:",
                    "   1. Add your Civitai API key above",
                    "   2. Enable 'auto_download' checkbox", 
                    "   3. Re-execute this node"
                ])
            
            log_lines.append("\n✨ Analysis complete!")
            
        except Exception as e:
            log_lines.extend([
                "",
                f"💥 Unexpected error: {str(e)}",
                "",
                "🔧 Debug info:",
                f"   • Input directory: {folder_paths.get_input_directory()}",
                f"   • Looking for: test.png",
                f"   • Full path: {os.path.join(folder_paths.get_input_directory(), 'test.png')}"
            ])
            logger.error(f"Error in CivitaiHelper: {str(e)}", exc_info=True)
        
        return (image_tensor, "\n".join(log_lines))
    
    def load_image_for_preview(self, image_path: str, log_lines: List[str]) -> torch.Tensor:
        """
        Load image for preview using ComfyUI's standard method
        """
        try:
            # Get file info
            file_size = os.path.getsize(image_path)
            log_lines.append(f"📏 File size: {format_file_size(file_size)}")
            
            # Load image using ComfyUI's method (similar to LoadImage)
            img = Image.open(image_path)
            log_lines.append(f"🖼️ Image: {img.format} {img.size} {img.mode}")
            
            # Handle EXIF orientation
            img = ImageOps.exif_transpose(img)
            
            # Convert to RGB
            if img.mode == 'I':
                img = img.point(lambda i: i * (1 / 255))
            image = img.convert("RGB")
            
            # Convert to tensor
            image_array = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array)[None,]
            
            log_lines.append(f"✅ Image loaded for preview: {image_tensor.shape}")
            
            return image_tensor
            
        except Exception as e:
            log_lines.append(f"❌ Error loading image for preview: {str(e)}")
            # Return black image as fallback
            return torch.zeros((1, 512, 512, 3), dtype=torch.float32)
    
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

    @classmethod
    def IS_CHANGED(cls, image):
        """
        Check if the image file has changed (similar to LoadImage)
        """
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()
    
    @classmethod
    def VALIDATE_INPUTS(cls, image):
        """
        Validate the input image file (similar to LoadImage)
        """
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True 