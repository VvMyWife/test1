"""PDF 流水线 item 流的 SSOT Pydantic 模型。

字段语义与各阶段说明见 ``docs/PDF_PIPELINE_SCHEMAS.md``。

大负载：超长 OCR 全文建议走 ``PageItem.text_artifact_ref`` 或外置存储；
若 ``DocumentResultItem.spans`` 内嵌海量命中，可能触发 BSON/HTTP 等单文档体积上限，
宜改为引用模式或独立 span 存储。
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --- 几何与分值（语义在类型侧写清）-----------------------------------------

# ``(x0, y0, x1, y1)``：``coord_space`` 下的轴对齐框（单位随坐标系），非 (x, y, w, h)。
BboxQuad = tuple[float, float, float, float]

ConfidenceScore = Annotated[float, Field(ge=0.0, le=1.0, description="置信度分值，闭区间 [0, 1]")]


class BoundingBox(BaseModel):
    """Daft/PyArrow 友好的文本框，使用左上角 + 宽高。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x: int = Field(..., description="左上角 x 坐标")
    y: int = Field(..., description="左上角 y 坐标")
    w: int = Field(..., ge=0, description="宽度")
    h: int = Field(..., ge=0, description="高度")


class TextBlock(BaseModel):
    """页面上的一个可独立处理文本块。

    字段名与 Daft document-processing 示例保持兼容：下游可直接访问
    ``text_blocks[].text`` 与 ``text_blocks[].bounding_box.{x,y,w,h}``。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(..., description="文本块内容")
    bounding_box: BoundingBox = Field(..., description="文本块边界框")
    block_type: str | None = Field(default=None, description="来源版面类型，如 text/title/page_number")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0, description="可选 OCR/版面置信度")
    meta: dict[str, Any] = Field(default_factory=dict, description="保留 MinerU text_level、source 等附加信息")


class TableCell(BaseModel):
    """Normalized cell-level table structure."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cell_id: str = Field(..., min_length=1)
    text: str = ""
    bounding_box: BoundingBox | None = None
    row_index: int | None = Field(default=None, ge=0)
    col_index: int | None = Field(default=None, ge=0)
    row_span: int = Field(default=1, ge=1)
    col_span: int = Field(default=1, ge=1)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    meta: dict[str, Any] = Field(default_factory=dict)


class TableBlock(BaseModel):
    """Table region with cell-level structure.

    Keep the recognizer HTML when available, while exposing cells as JSON so
    downstream operators do not need to parse HTML for row/column granularity.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    table_id: str = Field(..., min_length=1)
    page_index: int = Field(..., ge=0)
    provider: str = Field(default="paddleocr")
    bounding_box: BoundingBox | None = None
    coord_space: str = Field(default="mineru_layout")
    html: str | None = None
    cells: list[TableCell] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    meta: dict[str, Any] = Field(default_factory=dict)


class DPI(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    height: float
    width: float


class ImageSize(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    width: int
    height: int
    dpi: DPI | None = None


class ParsedPage(BaseModel):
    """Daft UDF 期望的页面级结构。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    page_index: int = Field(..., ge=0)
    text_blocks: list[TextBlock] = Field(default_factory=list)
    table_blocks: list[TableBlock] = Field(default_factory=list)
    image_size: ImageSize | None = None


class ParsedPdf(BaseModel):
    """Daft UDF 期望的文档级结构。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pdf_path: str | None = None
    total_pages: int = Field(..., ge=0)
    pages: list[ParsedPage] = Field(default_factory=list)


class CoordSpace(StrEnum):
    """``bbox`` 所使用的已登记坐标系。

    新增枚举取值须经跨模块评审；参见 ``docs/requirement/design/00_cross_module_contracts.md`` §3。
    """

    MINERU_LAYOUT = "mineru_layout"
    PDF_POINTS = "pdf_points"
    IMAGE_PIXELS = "image_pixels"
    NORMALIZED_0_1 = "normalized_0_1"


class ArtifactRef(BaseModel):
    """与跨模块 ``ArtifactRef`` 对齐的二进制或大 JSON 引用。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str = Field(..., min_length=1, description="产物类型，如 image_path、ocr_text、json、middle_json")
    uri: str | None = Field(default=None, description="外置载荷时的存储 URI")
    inline: Any | None = Field(default=None, description="未使用 uri 时的小体积内联载荷")
    checksum: str | None = Field(default=None, description="针对 uri 指向内容的可选完整性校验")
    meta: dict[str, Any] = Field(
        default_factory=dict,
        description="如 content_type、parser_version、page_count 等元信息",
    )

    @model_validator(mode="after")
    def exactly_one_payload(self) -> Self:
        has_uri = self.uri is not None and self.uri.strip() != ""
        has_inline = self.inline is not None
        if has_uri == has_inline:
            raise ValueError("Exactly one of uri (non-empty) or inline must be set")
        return self


class ProvenanceMin(BaseModel):
    """PII / 审计用最小溯源结构（字符偏移均为页内）。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(..., min_length=1, description="来源，如 mineru_span、ocr_line、user_selection")
    page_index: int = Field(..., ge=0, description="从 0 起的页码")
    char_start: int = Field(..., ge=0, description="页内起始偏移（全项目统一一种编码约定）")
    char_end: int = Field(
        ...,
        ge=0,
        description="页内结束偏移（开区间或闭区间由流水线 SSOT 另行约定）",
    )
    span_id: str | None = Field(default=None, description="有则填的稳定 span 标识")
    bbox: BboxQuad | None = Field(default=None, description="可选，与命中同一 coord_space 下的 (x0,y0,x1,y1)")
    quote: str | None = Field(default=None, description="供人工复核的短引用片段")

    @model_validator(mode="after")
    def char_end_ge_start(self) -> Self:
        if self.char_end < self.char_start:
            raise ValueError("char_end must be >= char_start")
        return self


# --- 流水线各阶段 item -------------------------------------------------------


class ArchiveJobItem(BaseModel):
    """控制面输入：针对一批档案（PDF 路径列表）的一次批处理任务。

    当前阶段为内部使用，不设租户维度；审计上强调**谁于何时触发**本批任务。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    archive_id: str = Field(..., min_length=1, description="业务档案 / 批次标识")
    job_id: str = Field(..., min_length=1, description="可回溯的稳定平台任务 ID")
    triggered_by_user_id: str = Field(
        ...,
        min_length=1,
        description="发起本次批处理的用户 ID（与账号体系对齐的稳定标识）",
    )
    triggered_at: datetime | None = Field(
        default=None,
        description="任务触发时间（建议 UTC；与审计时间线一致）",
    )
    source_paths: list[str] = Field(
        ...,
        min_length=1,
        description="待处理的 PDF 文件路径或 URI 列表",
    )
    options: dict[str, Any] = Field(default_factory=dict, description="解析器、OCR、规则等选项")
    config_version: str | None = Field(
        default=None,
        description="可选；若填写宜与 OperatorContext.config_version 保持一致",
    )

    @field_validator("source_paths")
    @classmethod
    def source_paths_non_empty_strings(cls, v: list[str]) -> list[str]:
        for i, p in enumerate(v):
            if not isinstance(p, str) or not p.strip():
                raise ValueError(f"source_paths[{i}] must be a non-empty string")
        return v


class DocumentItem(BaseModel):
    """文件级单元：单份 PDF（或逻辑文档），用于拆页、汇总与文档级审计。

    携带档案业务归属人与本次任务触发人，便于下游 item 无 join 审计。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    archive_id: str = Field(..., min_length=1, description="所属档案 ID")
    archive_owner_user_id: str = Field(
        ...,
        min_length=1,
        description="该档案在业务上的归属人用户 ID（owner）",
    )
    triggered_by_user_id: str = Field(
        ...,
        min_length=1,
        description="发起产生本文档所在批次的用户 ID（通常与 ArchiveJobItem.triggered_by_user_id 一致）",
    )
    doc_id: str = Field(
        ...,
        min_length=1,
        description="流水线生成的稳定文档 ID（如哈希或 UUID）",
    )
    file_uri: str = Field(
        ...,
        min_length=1,
        max_length=8192,
        description="源定位：本地路径、file://、s3://、https:// 等（会做 trim）",
    )
    mime_type: str = Field(default="application/pdf", description="MIME 类型，默认 PDF")
    file_hash: str | None = Field(default=None, description="内容哈希，用于幂等或缓存")
    num_pages: int | None = Field(
        default=None,
        ge=1,
        description="解析后填写；已知页数时须 ≥ 1",
    )
    meta: dict[str, Any] = Field(default_factory=dict, description="上传者、时间戳、解析器版本等")
    artifact_ref: ArtifactRef | None = Field(
        default=None,
        description="可选整文档旁路产物（如打包后的 layout bundle）",
    )

    @field_validator("file_uri")
    @classmethod
    def file_uri_trim_and_bounds(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("file_uri must not be empty or whitespace-only")
        if len(s) > 8192:
            raise ValueError("file_uri exceeds maximum length")
        return s


class PageItem(BaseModel):
    """主干 item：单页（从 0 起编页码），承载 OCR、版面、检测等阶段。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    archive_id: str = Field(
        ...,
        min_length=1,
        description="与 DocumentItem.archive_id 一致，便于分区聚合而无需回连 Document",
    )
    archive_owner_user_id: str = Field(
        ...,
        min_length=1,
        description="与 DocumentItem.archive_owner_user_id 一致，便于按归属人审计与分区",
    )
    triggered_by_user_id: str = Field(
        ...,
        min_length=1,
        description="与 DocumentItem.triggered_by_user_id 一致",
    )
    doc_id: str = Field(..., min_length=1, description="文档 ID")
    page_index: int = Field(..., ge=0, description="从 0 起的页索引，与 MinerU page_idx 对齐")
    text: str | None = Field(
        default=None,
        description="页内联全文；OCR 极大时建议改用 text_artifact_ref",
    )
    text_artifact_ref: ArtifactRef | None = Field(
        default=None,
        description="外置全文（如指向 .txt 的 uri），以降低流水线内存与带宽",
    )
    text_blocks: list[TextBlock] = Field(
        default_factory=list,
        description="Daft/批处理友好的页内文本块，保留 bbox 与阅读顺序",
    )
    table_blocks: list[TableBlock] = Field(default_factory=list)
    page_meta: dict[str, Any] = Field(
        default_factory=dict,
        description="如宽高、旋转、语言提示、layout_parser 等",
    )
    image_ref: ArtifactRef | None = Field(default=None, description="页光栅图或预览引用")
    layout_ref: ArtifactRef | None = Field(
        default=None,
        description="版面产物引用（如 MinerU middle.json 切片）",
    )


class SensitiveSpanItem(BaseModel):
    """派生命中：页面上的一段敏感文本；paragraph 可表示为 span_type=paragraph。

    审计上区分：档案归属、任务触发、本条命中的检测责任（人或系统自动）。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    archive_id: str = Field(
        ...,
        min_length=1,
        description="与父级档案一致，无 join 即可按 archive 分区统计",
    )
    archive_owner_user_id: str = Field(
        ...,
        min_length=1,
        description="档案业务归属人用户 ID，与 DocumentItem 一致",
    )
    triggered_by_user_id: str = Field(
        ...,
        min_length=1,
        description="批任务触发人用户 ID，与 ArchiveJobItem / DocumentItem 一致",
    )
    doc_id: str = Field(..., min_length=1, description="文档 ID")
    page_index: int = Field(..., ge=0, description="从 0 起的页索引")
    span_id: str = Field(..., min_length=1, description="用于审计与高亮对齐的稳定 ID")
    text: str = Field(..., description="命中片段或归一化后的文本")
    risk_type: str = Field(..., min_length=1, description="风险类别，如 id_number、phone、secret、contract_clause")
    score: ConfidenceScore = Field(..., description="风险分值，[0, 1]")
    bbox: BboxQuad | None = Field(
        default=None,
        description="coord_space 下的轴对齐 (x0,y0,x1,y1)，非 (x,y,w,h)",
    )
    coord_space: CoordSpace | None = Field(
        default=None,
        description="有 bbox 时必填；原点与单位由枚举取值定义",
    )
    evidence: dict[str, Any] = Field(default_factory=dict, description="规则或模型侧证据载荷")
    provenance: ProvenanceMin | None = Field(
        default=None,
        description="结构化页内偏移；PII 级审计路径建议填写",
    )
    span_type: str | None = Field(
        default=None,
        description="如 paragraph、line、table_cell，用于渲染与聚合",
    )
    detection_actor: Literal["system", "user"] = Field(
        default="system",
        description="system：纯自动检测；user：由用户显式发起的检测/重跑等",
    )
    detected_by_user_id: str | None = Field(
        default=None,
        description="对本次命中负责的用户 ID；自动检测可为 None，用户发起或人工标注时填写",
    )
    detected_at: datetime | None = Field(
        default=None,
        description="产生本命中记录的时间（建议 UTC）",
    )

    @model_validator(mode="after")
    def bbox_requires_coord_space(self) -> Self:
        if self.bbox is not None and self.coord_space is None:
            raise ValueError("coord_space is required when bbox is set")
        return self

    @model_validator(mode="after")
    def provenance_page_index_matches(self) -> Self:
        if self.provenance is not None and self.provenance.page_index != self.page_index:
            raise ValueError("provenance.page_index must match SensitiveSpanItem.page_index")
        return self

    @model_validator(mode="after")
    def user_initiated_detection_requires_actor(self) -> Self:
        if self.detection_actor == "user" and (
            self.detected_by_user_id is None or not self.detected_by_user_id.strip()
        ):
            raise ValueError("detection_actor=user requires non-empty detected_by_user_id")
        return self


class DocumentResultItem(BaseModel):
    """文档级汇总：供持久化与前端展示。

    注意：内嵌成千上万条 ``SensitiveSpanItem`` 可能超过单文档大小限制（如 MongoDB 16MB）。
    宜将 span 外置存储，仅在 ``summary`` / ``artifacts`` 中保留引用或聚合指标。

    人工终局动作（如「确认发布」）使用 ``confirmed_*``，满足审计对责任人与时刻的要求。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str = Field(..., min_length=1, description="文档 ID")
    archive_id: str = Field(..., min_length=1, description="档案 ID，跨文档聚合时使用")
    archive_owner_user_id: str = Field(
        ...,
        min_length=1,
        description="档案业务归属人用户 ID",
    )
    triggered_by_user_id: str = Field(
        ...,
        min_length=1,
        description="批任务触发人用户 ID",
    )
    confirmed_by_user_id: str | None = Field(
        default=None,
        description="最终确认结论的用户 ID（如人工定稿、发布前复核）",
    )
    confirmed_at: datetime | None = Field(
        default=None,
        description="确认动作发生时间（建议 UTC；与 confirmed_by_user_id 成对使用）",
    )
    summary: dict[str, Any] = Field(
        ...,
        description="总体风险等级、命中数、关键证据摘要等",
    )
    spans: list[SensitiveSpanItem] = Field(default_factory=list, description="可选内嵌命中列表")
    page_summaries: list[dict[str, Any]] = Field(
        default_factory=list,
        description="按页的统计或最强命中摘要",
    )
    artifacts: list[ArtifactRef] = Field(
        default_factory=list,
        description="结构化产物引用，如 middle_json、叠加渲染图等",
    )

    @model_validator(mode="after")
    def confirmation_user_and_time_paired(self) -> Self:
        has_user = self.confirmed_by_user_id is not None and self.confirmed_by_user_id.strip() != ""
        has_at = self.confirmed_at is not None
        if has_user != has_at:
            raise ValueError(
                "confirmed_by_user_id and confirmed_at must both be set or both omitted (audit pair)"
            )
        return self

    @model_validator(mode="after")
    def embedded_spans_match_document(self) -> Self:
        for s in self.spans:
            if s.doc_id != self.doc_id:
                raise ValueError(
                    f"Embedded span doc_id {s.doc_id!r} must match DocumentResultItem.doc_id {self.doc_id!r}"
                )
            if s.archive_id != self.archive_id:
                raise ValueError(
                    f"Embedded span archive_id {s.archive_id!r} must match "
                    f"DocumentResultItem.archive_id {self.archive_id!r}"
                )
            if s.archive_owner_user_id != self.archive_owner_user_id:
                raise ValueError(
                    f"Embedded span archive_owner_user_id {s.archive_owner_user_id!r} must match "
                    f"DocumentResultItem.archive_owner_user_id {self.archive_owner_user_id!r}"
                )
            if s.triggered_by_user_id != self.triggered_by_user_id:
                raise ValueError(
                    f"Embedded span triggered_by_user_id {s.triggered_by_user_id!r} must match "
                    f"DocumentResultItem.triggered_by_user_id {self.triggered_by_user_id!r}"
                )
        return self
