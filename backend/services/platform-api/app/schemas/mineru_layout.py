from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MinerULayoutExtractRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_uri: str = Field(..., min_length=1)
    output_dir: str = Field(..., min_length=1)
    source_file_name: str | None = None
    mineru_options: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float | None = Field(default=None, gt=0)


class MinerULayoutPathRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pdf_path: str = Field(..., min_length=1)
    output_dir: str = Field(..., min_length=1)
    mineru_options: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float | None = Field(default=None, gt=0)


class MinerULayoutBatchDocumentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_uri: str = Field(..., min_length=1)
    source_file_name: str = Field(..., min_length=1)


class MinerULayoutBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    documents: list[MinerULayoutBatchDocumentInput] = Field(..., min_length=1)
    output_dir: str = Field(..., min_length=1)
    mineru_options: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float | None = Field(default=None, gt=0)


class MinerUGeneratedJson(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_pdf: str
    source_file_name: str
    success: bool = True
    json_path: str
    artifact_dir: str
    page_count: int = Field(ge=0)
    text_block_count: int = Field(ge=0)
    table_block_count: int = Field(ge=0)
    elapsed_seconds: float = Field(ge=0)


class MinerULayoutBatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_count: int = Field(ge=0)
    output_dir: str
    documents: list[MinerUGeneratedJson] = Field(default_factory=list)
