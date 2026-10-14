from typing import List, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ..schema import ModelDetail, SearchPage, SourceInfo

DEFAULT_LIMIT = 24
MAX_LIMIT = 50
MAX_QUERY_LENGTH = 8192


class SearchQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    q: str = Field("", max_length=MAX_QUERY_LENGTH)
    types: List[str] = Field(default_factory=list)
    base_models: List[str] = Field(default_factory=list)
    sort: Optional[str] = None
    period: Optional[str] = None
    cursor: Optional[str] = None
    limit: int = Field(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT)
    safe: bool = False


@runtime_checkable
class Source(Protocol):
    id: str
    label: str

    async def info(self) -> SourceInfo:
        ...

    async def search(self, query: SearchQuery, *, max_level: Optional[int] = None) -> SearchPage:
        ...

    async def detail(self, model_id: str, *, max_level: Optional[int] = None) -> ModelDetail:
        ...


class UnknownSource(LookupError):
    def __init__(self, source_id: str) -> None:
        super().__init__(f"Unknown source {source_id!r}")
        self.source_id = source_id
