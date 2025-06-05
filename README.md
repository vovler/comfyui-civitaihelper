# ComfyUI Civitai Helper Extension

A powerful ComfyUI custom node collection that automatically parses PNG workflow images from Civitai, extracts embedded ComfyUI metadata, identifies missing models, and downloads them directly using the Civitai API.

## Features

✨ **PNG Workflow Parsing**: Automatically extracts ComfyUI workflow JSON from PNG images downloaded from Civitai  
🖱️ **Drag & Drop Support**: Simply drag and drop Civitai PNG images directly into ComfyUI  
🔍 **Smart Model Detection**: Identifies missing checkpoints, LoRAs, VAE, embeddings, ControlNet, and other models  
🚀 **Automatic Downloads**: Downloads missing models directly from Civitai using their API  
📁 **Intelligent Organization**: Saves models to the correct ComfyUI folders automatically  
🔧 **Advanced Matching**: Uses fuzzy matching to find models even with slight name variations  
⚡ **Progress Tracking**: Real-time download progress with file size information  
🛡️ **Backup Support**: Optional backup creation before overwriting existing files  
🎯 **Multi-format Support**: Prioritizes .safetensors files but supports .pt, .ckpt, and other formats  
⚙️ **Persistent Settings**: Save your API key and preferences for easy reuse  
🎛️ **Separate Download Node**: Dedicated node for downloading with safety controls  

## Nodes Overview

### 🎨 Civitai Workflow Parser
Main node for analyzing Civitai PNG images and identifying missing models:
- **Drag & Drop Support**: Simply drag PNG images into the node
- **Workflow Extraction**: Automatically extracts ComfyUI workflow from image metadata
- **Model Detection**: Identifies all missing models in the workflow
- **Check Mode**: Safe analysis without downloads

### 📥 Civitai Model Downloader  
Dedicated download node with safety controls:
- **Safe Downloads**: Requires explicit confirmation to download
- **Batch Processing**: Downloads multiple models efficiently
- **Progress Tracking**: Real-time download status and progress
- **Error Handling**: Detailed error reporting and recovery

### ⚙️ Civitai Settings
Settings management node:
- **API Key Storage**: Securely save your Civitai API key
- **Persistent Settings**: Preferences saved between sessions
- **Settings Export/Import**: Backup and restore your configuration

## Installation

1. **Clone this repository** into your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes/
   git clone https://github.com/your-username/comfui-civitaihelper.git
   ```

2. **Install dependencies**:
   ```bash
   cd comfui-civitaihelper
   pip install -r requirements.txt
   ```

3. **Restart ComfyUI**

## Quick Start Guide

### Step 1: Set Up Your API Key

1. Get your Civitai API key from [Civitai Account Settings](https://civitai.com/user/account)
2. Add the **⚙️ Civitai Settings** node to your workflow
3. Set action to "Save New Key" and enter your API key
4. Execute the node to save your key

### Step 2: Analyze a Civitai Image

1. Add the **🎨 Civitai Workflow Parser** node
2. **Drag and drop** a PNG image from Civitai directly onto the image input
3. Set `check_only=True` for safe analysis
4. Execute to see workflow and missing models

### Step 3: Download Missing Models (Optional)

**Option A: Direct Download**
- Set `check_only=False` and `auto_download=True` in the parser node

**Option B: Separate Download Node (Recommended)**
1. Add the **📥 Civitai Model Downloader** node
2. Connect the workflow_json and missing_models outputs from the parser
3. Set `download_models=True` to confirm downloads
4. Execute to download missing models

## Detailed Usage

### Drag & Drop Workflow

The easiest way to use the extension:

1. **Find a Civitai Image**: Browse [Civitai.com](https://civitai.com) and find an image you like
2. **Save the Image**: Right-click and save the PNG image (includes workflow metadata)
3. **Drag & Drop**: Drag the saved PNG directly onto the image input of the parser node
4. **Analyze**: Execute the node to extract the workflow and identify missing models
5. **Download**: Use the download node to automatically download missing models

### Manual Path Input

You can also specify image paths manually:
- Leave the image input empty
- Enter the path in the `image_path` field
- Supports absolute and relative paths

### Settings Management

The **⚙️ Civitai Settings** node provides several actions:

- **Get Current Key**: Check if an API key is saved (shows masked version)
- **Save New Key**: Store a new API key securely  
- **Clear Saved Key**: Remove the saved API key

### Safety Features

- **Check Mode**: Default safe mode that only analyzes without downloading
- **Download Confirmation**: Dedicated download node requires explicit confirmation
- **Backup Creation**: Optional backup of existing files before overwriting
- **Progress Monitoring**: Real-time download progress and status

## Supported Model Types

- **Checkpoints**: Stable Diffusion models (.safetensors, .ckpt, .pt)
- **LoRAs**: Low-Rank Adaptation models (.safetensors, .pt)
- **VAE**: Variational Autoencoders (.safetensors, .pt, .ckpt)
- **Embeddings**: Textual Inversions (.safetensors, .pt, .bin)
- **ControlNet**: Control models (.safetensors, .pt)
- **CLIP Vision**: Vision models (.safetensors, .pt)
- **Upscale Models**: Super-resolution models (.pth, .pt, .safetensors)

## Model Organization

Models are automatically saved to the appropriate ComfyUI directories:

```
ComfyUI/models/
├── checkpoints/          # SD checkpoints
├── loras/                # LoRA models
├── vae/                  # VAE models
├── embeddings/           # Textual inversions
├── controlnet/           # ControlNet models
├── clip_vision/          # CLIP vision models
└── upscale_models/       # Upscaling models
```

## Advanced Configuration

### Settings File Location

Settings are automatically saved to:
- `ComfyUI/custom_nodes/civitai_helper_settings.json` (preferred)
- `~/.comfyui_civitai_settings.json` (fallback)

### Environment Variables

You can also use environment variables:

```bash
export CIVITAI_API_KEY="your_api_key_here"
export COMFYUI_MODELS_PATH="/path/to/your/models"
```

### Custom Configuration

Edit `config.py` to customize:
- API endpoints and timeouts
- Model type mappings  
- File extensions and priorities
- Download chunk sizes
- Search parameters

## Workflow Examples

### Basic Analysis Workflow
```
Image → 🎨 Civitai Workflow Parser → View Results
```

### Safe Download Workflow  
```
Image → 🎨 Civitai Workflow Parser → 📥 Civitai Model Downloader → Download
```

### Settings Management
```
⚙️ Civitai Settings → Save API Key → Ready to Use
```

## Troubleshooting

### Common Issues

**"No workflow metadata found in image"**
- Ensure the PNG was downloaded from Civitai with workflow metadata
- Try different PNG images from Civitai
- Check that the image hasn't been recompressed or edited

**"Please drag & drop an image or provide a valid image path"**
- Make sure you've either dragged an image onto the image input socket (not the node itself)
- Try using the image_path field as an alternative
- Restart ComfyUI if drag & drop stops working

**"Civitai API key is required"**
- Use the ⚙️ Civitai Settings node to save your API key
- Get your API key from [Civitai Account Settings](https://civitai.com/user/account)
- Ensure the key has proper permissions

**"Model not found on Civitai"**
- The model might not be available on Civitai
- Try searching manually on Civitai.com
- The model name might be slightly different

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

### Settings Reset

To reset all settings:
1. Use the ⚙️ Civitai Settings node with "Clear Saved Key"
2. Or manually delete the settings file

## Node Reference

### 🎨 Civitai Workflow Parser

**Inputs:**
- `image` (IMAGE, optional): Drag & drop image input
- `image_path` (STRING, optional): Manual path to image file
- `auto_download` (BOOLEAN): Enable automatic downloads
- `check_only` (BOOLEAN): Safe mode - analyze only
- `civitai_api_key` (STRING, optional): API key override
- `comfyui_models_path` (STRING, optional): Custom models path
- `prefer_safetensors` (BOOLEAN): Prefer .safetensors format
- `create_backup` (BOOLEAN): Backup existing files

**Outputs:**
- `workflow_json` (STRING): Extracted workflow JSON
- `missing_models` (STRING): List of missing models  
- `download_status` (STRING): Status and progress info
- `model_info` (STRING): Summary of all models

### 📥 Civitai Model Downloader

**Inputs:**
- `workflow_json` (STRING): Workflow from parser node
- `missing_models` (STRING): Missing models from parser node
- `download_models` (BOOLEAN): Confirmation to download
- `civitai_api_key` (STRING, optional): API key override
- `comfyui_models_path` (STRING, optional): Custom models path
- `prefer_safetensors` (BOOLEAN): Prefer .safetensors format
- `create_backup` (BOOLEAN): Backup existing files

**Outputs:**
- `download_status` (STRING): Download results summary
- `download_summary` (STRING): Detailed download information

### ⚙️ Civitai Settings

**Inputs:**
- `action` (CHOICE): Get Current Key | Save New Key | Clear Saved Key
- `new_api_key` (STRING, optional): New API key to save

**Outputs:**
- `status` (STRING): Operation result
- `current_key_info` (STRING): Current key information (masked)

## API Rate Limits

- Civitai has API rate limits
- The extension respects these limits
- If you hit rate limits, wait a few minutes before retrying

## Security Considerations

- API keys are stored locally in ComfyUI's directory
- Keys are masked in outputs for security
- Never share API keys publicly
- Regularly rotate your API keys

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Changelog

### v2.0.0
- 🆕 Drag & drop image support
- 🆕 Dedicated download node with safety controls  
- 🆕 Persistent settings management
- 🆕 Three separate nodes for different functions
- ✨ Enhanced UI with emojis and better feedback
- 🛡️ Improved security and error handling
- 🎯 Better separation of concerns

### v1.0.0
- Initial release
- PNG workflow parsing
- Model detection and downloading
- Support for all major model types
- Fuzzy model name matching
- Progress tracking
- Backup support

## Acknowledgments

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - The amazing UI for Stable Diffusion
- [Civitai](https://civitai.com) - The model sharing platform
- The open-source AI community

## Support

- Create an issue on GitHub for bugs or feature requests
- Join the ComfyUI community for general support
- Check Civitai documentation for API-related questions

---

**Note**: This extension requires an active internet connection and a valid Civitai API key to download models. Always respect the license terms of downloaded models.
