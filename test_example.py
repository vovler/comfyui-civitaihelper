#!/usr/bin/env python3
"""
Example test script for the ComfyUI Civitai Helper Extension

This script demonstrates how to use the extension programmatically.
For normal usage, you would use the ComfyUI node interface.
"""

import json
import os
from civitai_helper import CivitaiWorkflowParser

def test_civitai_helper():
    """
    Test the Civitai Helper functionality
    """
    # Initialize the parser
    parser = CivitaiWorkflowParser()
    
    # Example usage - you would need to replace these with actual values
    image_path = "path/to/your/civitai_workflow_image.png"
    api_key = "your_civitai_api_key_here"
    
    print("🚀 ComfyUI Civitai Helper Test")
    print("=" * 50)
    
    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        print("📝 Please update the image_path variable with a valid PNG file from Civitai")
        return
    
    # Test workflow extraction only (no downloads)
    print("🔍 Testing workflow extraction (check only mode)...")
    
    try:
        result = parser.process_civitai_image(
            image_path=image_path,
            civitai_api_key=api_key,
            auto_download=False,
            check_only=True
        )
        
        workflow_json, missing_models, download_status, model_info = result
        
        print("\n📊 Results:")
        print("-" * 30)
        
        if workflow_json:
            print("✅ Successfully extracted workflow from PNG")
            workflow = json.loads(workflow_json)
            print(f"📋 Workflow contains {len(workflow.get('nodes', []))} nodes")
        else:
            print("❌ Failed to extract workflow")
            return
        
        if missing_models:
            missing = json.loads(missing_models)
            print(f"🔍 Found {len(missing)} missing models")
            for model in missing:
                print(f"   • {model['name']} ({model['type']})")
        else:
            print("✅ All models are available locally")
        
        print(f"\n📈 Status: {download_status}")
        
        if model_info:
            print(f"\n📋 Model Summary:")
            print(model_info)
        
    except Exception as e:
        print(f"❌ Error during test: {str(e)}")
        return
    
    # Test with downloads (if API key is provided and there are missing models)
    if api_key and api_key != "your_civitai_api_key_here":
        print("\n" + "=" * 50)
        print("🚀 Testing with automatic downloads...")
        
        try:
            result = parser.process_civitai_image(
                image_path=image_path,
                civitai_api_key=api_key,
                auto_download=True,
                check_only=False,
                prefer_safetensors=True,
                create_backup=False
            )
            
            workflow_json, missing_models, download_status, model_info = result
            
            print(f"\n📥 Download Results:")
            print("-" * 30)
            print(download_status)
            
        except Exception as e:
            print(f"❌ Error during download test: {str(e)}")
    else:
        print("\n💡 Tip: Set a valid API key to test automatic downloads")
    
    print("\n✨ Test completed!")

def create_sample_config():
    """
    Create a sample configuration file
    """
    sample_config = """
# Sample configuration for testing
# Copy this to a .env file or set as environment variables

CIVITAI_API_KEY=your_api_key_here
COMFYUI_MODELS_PATH=/path/to/your/ComfyUI/models
TEST_IMAGE_PATH=/path/to/test/civitai_workflow.png

# Optional settings
PREFER_SAFETENSORS=true
CREATE_BACKUP=false
DEBUG_MODE=false
"""
    
    with open("sample_config.txt", "w") as f:
        f.write(sample_config)
    
    print("📝 Created sample_config.txt - update with your settings")

if __name__ == "__main__":
    print("ComfyUI Civitai Helper - Test Script")
    print("This script is for testing purposes only.")
    print("For normal usage, use the ComfyUI node interface.\n")
    
    # Create sample config
    create_sample_config()
    
    # Run the test
    test_civitai_helper() 