from typing import Dict, FrozenSet, Iterable, Optional

CIVITAI_TYPES: Dict[str, str] = {
    "Checkpoint": "checkpoint",
    "LORA": "lora",
    "LoCon": "lora",
    "DoRA": "lora",
    "TextualInversion": "embedding",
    "Controlnet": "controlnet",
    "Upscaler": "upscaler",
    "VAE": "vae",
    "Detection": "detection_bbox",
    "TextEncoder": "text_encoder",
    "CLIP": "text_encoder",
    "UNet": "unet",
    "LLM": "llm",
}

_CIVITAI_TYPES_FOLDED = {key.lower(): value for key, value in CIVITAI_TYPES.items()}


def civitai_mb_type(civitai_type: Optional[str]) -> Optional[str]:
    if not isinstance(civitai_type, str):
        return None
    return CIVITAI_TYPES.get(civitai_type) or _CIVITAI_TYPES_FOLDED.get(civitai_type.strip().lower())


WEIGHT_EXTENSIONS = (".safetensors", ".sft", ".gguf", ".ckpt", ".pt", ".pth", ".bin", ".onnx")
_CHECKPOINT_EXTENSIONS = (".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".bin")
UPSCALER_HINTS = ("esrgan", "upscal", "super-resolution")
HF_TEXT_OUTPUT_TASKS: FrozenSet[str] = frozenset({
    "any-to-any",
    "audio-classification",
    "audio-text-to-text",
    "automatic-speech-recognition",
    "conversational",
    "document-question-answering",
    "feature-extraction",
    "fill-mask",
    "graph-ml",
    "image-classification",
    "image-feature-extraction",
    "image-text-to-text",
    "image-to-text",
    "multiple-choice",
    "question-answering",
    "reinforcement-learning",
    "robotics",
    "sentence-similarity",
    "summarization",
    "table-question-answering",
    "table-to-text",
    "tabular-classification",
    "tabular-regression",
    "tabular-to-text",
    "text-classification",
    "text-generation",
    "text-ranking",
    "text-retrieval",
    "text2text-generation",
    "time-series-forecasting",
    "token-classification",
    "translation",
    "video-classification",
    "video-text-to-text",
    "visual-document-retrieval",
    "visual-question-answering",
    "zero-shot-classification",
    "zero-shot-image-classification",
    "zero-shot-object-detection",
})
HUGGINGFACE_MB_TYPES = frozenset({"lora", "controlnet", "vae", "upscaler", "diffusion_model", "checkpoint"})


def _has_upscaler_hint(tags: Iterable[str], repo_id: Optional[str]) -> bool:
    texts = [*tags, repo_id or ""]
    for text in texts:
        folded = text.lower().replace("_", "-")
        if any(hint in folded for hint in UPSCALER_HINTS):
            return True
    return False


def huggingface_mb_type(
    tags: Optional[Iterable[str]],
    pipeline_tag: Optional[str],
    filenames: Optional[Iterable[str]],
    repo_id: Optional[str] = None,
) -> Optional[str]:
    tag_list = [tag for tag in tags or () if isinstance(tag, str)]
    tag_set = {tag.strip().lower() for tag in tag_list}
    pipeline = pipeline_tag.strip().lower() if isinstance(pipeline_tag, str) else ""
    if pipeline in HF_TEXT_OUTPUT_TASKS:
        return None
    if "lora" in tag_set:
        return "lora"
    if "controlnet" in tag_set:
        return "controlnet"
    if "vae" in tag_set:
        return "vae"
    if _has_upscaler_hint(tag_list, repo_id if isinstance(repo_id, str) else None):
        return "upscaler"
    if "gguf" in tag_set:
        return "diffusion_model"
    if pipeline == "text-to-image":
        for name in filenames or ():
            if isinstance(name, str) and "/" not in name and name.lower().endswith(_CHECKPOINT_EXTENSIONS):
                return "checkpoint"
    return None
