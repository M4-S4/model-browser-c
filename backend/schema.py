from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

SourceId = Literal["civitai", "huggingface", "openmodeldb", "civarchive"]
NsfwCeiling = Literal["allowed", "blur", "blocked"]
NsfwMode = Literal["show", "blur", "hide"]


class NsfwContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    ceiling: NsfwCeiling
    mode: NsfwMode
    allowed_modes: Tuple[NsfwMode, ...]
    max_level: int = 2
    hide_poi: bool = False


class Image(BaseModel):
    url: Optional[str] = None
    blurhash: Optional[str] = None
    nsfw: bool = False
    nsfw_level: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    kind: Literal["image", "video"] = "image"
    poi: bool = False
    id: Optional[str] = None


class File(BaseModel):
    id: str
    name: str
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None
    format: Optional[str] = None
    precision: Optional[str] = None
    file_type: Optional[str] = None
    primary: bool = False
    download_url: str
    auth_provider_id: Optional[str] = None


class Version(BaseModel):
    id: str
    name: str
    base_model: Optional[str] = None
    trigger_words: List[str] = Field(default_factory=list)
    created_at: Optional[str] = None
    early_access: bool = False
    nsfw: bool = False
    images: List[Image] = Field(default_factory=list)
    files: List[File] = Field(default_factory=list)
    partial: bool = False

    @model_validator(mode="after")
    def _partial_has_no_files_or_images(self) -> "Version":
        if self.partial and (self.files or self.images):
            raise ValueError("a partial version carries no files or images")
        return self


class License(BaseModel):
    summary: str
    commercial_use: Optional[List[str]] = None
    no_credit: Optional[bool] = None
    derivatives: Optional[bool] = None
    different_license: Optional[bool] = None
    url: Optional[str] = None


class Stats(BaseModel):
    downloads: Optional[int] = None
    likes: Optional[int] = None
    rating: Optional[float] = None


class SearchItem(BaseModel):
    source: SourceId
    id: str
    name: str
    type: str
    mb_type: Optional[str] = None
    base_model: Optional[str] = None
    base_models: List[str] = Field(default_factory=list)
    creator: Optional[str] = None
    nsfw: bool = False
    poi: bool = False
    early_access: bool = False
    gated: bool = False
    thumbnail: Optional[Image] = None
    stats: Stats = Field(default_factory=Stats)
    source_url: str
    primary_sha256: Optional[str] = None
    installed: Optional[bool] = None


class SearchPage(BaseModel):
    source: SourceId
    items: List[SearchItem]
    next_cursor: Optional[str] = None


class ModelDetail(BaseModel):
    source: SourceId
    id: str
    name: str
    type: str
    mb_type: Optional[str] = None
    creator: Optional[str] = None
    description_html: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    license: Optional[License] = None
    source_url: str
    nsfw: bool = False
    poi: bool = False
    gated: bool = False
    stats: Stats = Field(default_factory=Stats)
    versions: List[Version] = Field(default_factory=list)


class FilterOption(BaseModel):
    value: str
    label: str


class SourceInfo(BaseModel):
    id: SourceId
    label: str
    types: List[FilterOption] = Field(default_factory=list)
    base_models: List[FilterOption] = Field(default_factory=list)
    sorts: List[FilterOption] = Field(default_factory=list)
    periods: List[FilterOption] = Field(default_factory=list)
    search_help: Optional[str] = None
    types_label: str = "Types"
    base_models_label: str = "Base model"
    base_model_label: str = "Base model"
    notice: Optional[str] = None
