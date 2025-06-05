from .civitai_helper import CivitaiWorkflowParser, CivitaiModelDownloader, CivitaiSettingsNode

NODE_CLASS_MAPPINGS = {
    "CivitaiWorkflowParser": CivitaiWorkflowParser,
    "CivitaiModelDownloader": CivitaiModelDownloader,
    "CivitaiSettingsNode": CivitaiSettingsNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CivitaiWorkflowParser": "🎨 Civitai Workflow Parser",
    "CivitaiModelDownloader": "📥 Civitai Model Downloader", 
    "CivitaiSettingsNode": "⚙️ Civitai Settings"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] 