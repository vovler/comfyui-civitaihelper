from .civitai_helper import CivitaiWorkflowParser, CivitaiModelDownloader, CivitaiSettingsNode, CivitaiImageHandler, ShowText

NODE_CLASS_MAPPINGS = {
    "🎨 Civitai Workflow Parser": CivitaiWorkflowParser,
    "📥 Civitai Model Downloader": CivitaiModelDownloader,
    "⚙️ Civitai Settings": CivitaiSettingsNode,
    "🖼️ Civitai Image Handler": CivitaiImageHandler,
    "📝 Show Text": ShowText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "🎨 Civitai Workflow Parser": "🎨 Civitai Workflow Parser",
    "📥 Civitai Model Downloader": "📥 Civitai Model Downloader",
    "⚙️ Civitai Settings": "⚙️ Civitai Settings",
    "🖼️ Civitai Image Handler": "🖼️ Civitai Image Handler",
    "📝 Show Text": "📝 Show Text",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] 