# PDF Pipeline Schemas (SSOT)

本文档定义面向 **PDF 批处理 / 审查** 的 **SSOT（Single Source of Truth）item schemas**，用于 `platform_foundation` 的算子（operator）链路编排与 `input_schema()/output_schema()` 契约声明。

- **算子契约**：`backend/foundation/platform_foundation/operators/contracts.py`（`OperatorSchema.item(BaseModel)`）、`backend/foundation/platform_foundation/operators/base.py`（`can_accept` 的硬匹配规则）。  
- **算子总览**：`docs/OPERATORS.md`（算子与 pipeline 的职责边界、重试语义、可观测性）。  
- **跨模块约定**：`docs/requirement/design/00_cross_module_contracts.md`（`page_index`、`bbox`、`coord_space`、provenance 最小字段、`ArtifactRef`）。  
- **LLM 推理封装**：`docs/INFERENCE.md`（OpenAI SDK 优先、PromptInputsModel、结构化输出、审计/缓存/快照）。  

---

## 设计目标

- **契约清晰**：每个阶段输入/输出的数据结构明确，避免“靠 dict 约定字段”的隐式耦合。
- **Fail-fast**：在编排阶段通过 schema 兼容性检查尽早发现链路接错（而不是跑到下游才报错）。
- **可审计**：所有派生结果（敏感段落/命中/证据）都能回溯到 `doc_id/page_index/span_id` 与 provenance；**内部使用阶段**以**用户与时刻**为主（档案归属、任务触发、检测责任、人工确认），不设租户字段。

---

## 关键约束（来自现有框架实现）

### `OperatorSchema` 的“硬匹配”

`BaseOperator.can_accept()` 当前规则是：

- `expected.kind == upstream.kind`
- 且 `expected.schema == upstream.schema`

这意味着：**上下游的 item schema 必须引用同一个 Pydantic `BaseModel` 类型**（SSOT），否则即便字段“看起来一样”也可能被判定不兼容。

### 坐标与页号约定（必须对齐）

依据 `docs/requirement/design/00_cross_module_contracts.md`：

- **`page_index`**：0-based，与 MinerU `page_idx` 对齐。
- **`bbox`**：优先使用 `[x0, y0, x1, y1]`（float[4]），并与 `coord_space` 一致。
- **`coord_space`**：新增取值需评审；文档中应显式列出支持的 `coord_space` 及互转规则。

### Provenance（溯源）最小字段（建议稳定）

建议所有 `SensitiveSpan`/evidence 结构至少具备：

- `source`（如 `mineru_span` / `ocr_line` / `user_selection`）
- `page_index`
- `char_start` / `char_end`（页内偏移；全项目统一一种编码偏移体系）
- 可选：`span_id`、`bbox`、`quote`

---

## SSOT Schemas（Pydantic item models）

> 本节描述的是**权威字段清单与语义**。实际代码落地建议放在：  
> `backend/foundation/platform_foundation/contracts/pdf_items.py`（避免在各个 operator 文件里重复定义 schema）。

**配套类型（与 item 同级导出）**：`CoordSpace`（bbox 坐标系枚举）、`ArtifactRef`（产物引用）、`ProvenanceMin`（页内字符溯源最小集）、`BboxQuad` / `ConfidenceScore`（类型别名）。

### 1) `ArchiveJobItem`（控制面 / 任务输入）

**用途**：表达一次批处理任务的输入与选项（“要处理哪些 PDF、用什么策略”），一般只作为 pipeline 的入口，不做大规模逐页处理的主干数据结构。

- **必填**
  - `archive_id`: str（业务侧的档案/批次 ID）
  - `job_id`: str（平台任务 ID）
  - `triggered_by_user_id`: str（**谁发起**本次批处理；与内部账号体系的用户 ID 对齐）
  - `source_paths`: list[str]（PDF 文件路径或 URI 列表）
- **可选**
  - `triggered_at`: datetime（**何时发起**；建议 UTC，写入审计时间线）
  - `options`: dict[str, Any]（解析器选择、OCR 开关、规则集版本等）
  - `config_version`: str（也可放在 `OperatorContext.config_version`，二者选一并保持一致）

**ID 规则**：`job_id` 必须稳定可回溯；`archive_id` 用于跨 doc 聚合。

---

### 2) `DocumentItem`（文件级）

**用途**：表示一个 PDF 文件（或一个“逻辑文档”）的最小处理单元，用于拆页、汇总与文档级审计。

- **必填**
  - `archive_id`: str
  - `archive_owner_user_id`: str（**档案业务归属人**，owner）
  - `triggered_by_user_id`: str（产生本文档所在批次的**任务触发人**，通常与 `ArchiveJobItem.triggered_by_user_id` 一致）
  - `doc_id`: str（建议由 pipeline 生成：例如 `sha256(archive_id + file_uri + payload_hash)` 的短形式或 UUID）
  - `file_uri`: str（源文件定位；本地路径 / 对象存储 URI）
  - `mime_type`: str（如 `application/pdf`）
- **可选**
  - `file_hash`: str（用于幂等/缓存；可选）
  - `num_pages`: int（解析后填充；**若设置则必须 ≥ 1**）
  - `meta`: dict[str, Any]（上传者、采集时间、解析器版本等）
  - `artifact_ref`: `ArtifactRef`（`kind` + `uri` 与 `inline` 二选一；见代码 `pdf_items.ArtifactRef`）

**注意**：`file_uri` 可能包含敏感路径信息，日志需谨慎。

---

### 3) `PageItem`（主干 item：推荐以 Page 为并行粒度）

**用途**：PDF pipeline 的主干数据结构。绝大多数 OCR、版面、规则扫描、敏感检测都可以以 page 为并行粒度。

- **必填**
  - `archive_id`: str（与 `DocumentItem.archive_id` 一致，便于 MapReduce/OLAP 分区而无需回连 Document）
  - `archive_owner_user_id`: str（与 `DocumentItem.archive_owner_user_id` 一致）
  - `triggered_by_user_id`: str（与 `DocumentItem.triggered_by_user_id` 一致）
  - `doc_id`: str
  - `page_index`: int（0-based）
- **可选（按阶段逐步填充）**
  - `text`: str（页文本：来自 OCR 或 PDF text extraction；大文本可改用 `text_artifact_ref`）
  - `text_artifact_ref`: `ArtifactRef`（外置全文，降低流水线内存/带宽）
  - `page_meta`: dict[str, Any]
    - 可包含：`width/height`、`rotation`、`language_hint`、`layout_parser` 等
  - `image_ref` / `layout_ref`: `ArtifactRef`（页图、MinerU middle.json 等）

**约定**：若 `text` 为空，后续检测算子应明确策略（跳过/报错/标记不可读）。

#### 下一版本演进目标：`payloads`（Typed Payload 分层，避免 PageItem 膨胀）

当前版本保持显式字段（`text`/`text_artifact_ref`/`layout_ref`/`image_ref`）为主，便于快速落地与清晰读取。

随着解析器与产物类型增多（多 OCR 引擎、多 layout 解析器、不同粒度的文本对齐产物），建议在下一版本引入可扩展 payload 容器：

- `payloads: dict[str, ArtifactRef]`

并采用“类型 + 解析器 + 版本”的 key 命名约定（示例）：

- `text.ocr.v1`
- `text.extracted_pdf.v1`
- `layout.mineru.v1`
- `layout.hybrid.v1`
- `image.page_png.v1`

与当前显式字段的映射建议：

- `text_artifact_ref` ≈ `payloads["text.ocr.v1"]`（或默认 text key）
- `layout_ref` ≈ `payloads["layout.mineru.v1"]`
- `image_ref` ≈ `payloads["image.page_png.v1"]`

收益：

- 避免 PageItem 变成 God Object
- 支持多解析器并存与灰度切换
- LLM 可按需拉取 payload（显著降低 token 成本）

---

### 4) `SensitiveSpanItem`（派生结果：敏感段落/片段）

**用途**：表达“在某页某范围内发现的敏感片段”，可用于高亮、审计、下游索引与汇总。

- **必填**
  - `archive_id`: str（与 Document 一致，便于聚合）
  - `archive_owner_user_id`: str（档案归属人，与 Document 一致）
  - `triggered_by_user_id`: str（批任务触发人，与 Job/Document 一致）
  - `doc_id`: str
  - `page_index`: int（0-based）
  - `span_id`: str（稳定 ID；若上游无 ID，建议由 pipeline 生成）
  - `text`: str（命中文本或归一化后的片段）
  - `risk_type`: str（分类，如 `id_number` / `phone` / `secret` / `contract_clause`）
  - `score`: float（**闭区间 [0, 1]**，类型别名为 `ConfidenceScore`）
- **可选**
  - `bbox`: **四元组** `(x0, y0, x1, y1)` 浮点（轴对齐框，**非** `(x, y, w, h)`）；与 `coord_space` 单位一致（实现中为 tuple，而非 list）
  - `coord_space`: `CoordSpace` 枚举（如 `mineru_layout`、`pdf_points`、`image_pixels`、`normalized_0_1`）；**有 bbox 时必填**
  - `evidence`: dict[str, Any]（规则命中、模型证据、prompt/response 摘要等）
  - `provenance`: `ProvenanceMin`（结构化 `source` / `page_index` / `char_start` / `char_end` 等；`page_index` 须与本条 span 的 `page_index` 一致）
  - `span_type`: str（可选：`paragraph`/`line`/`table_cell`；用于渲染/聚合）
  - `detection_actor`: `system` | `user`（自动 vs 用户发起的检测/重跑）
  - `detected_by_user_id`: str | None（**本条命中责任用户**；纯自动可为 `None`；`detection_actor=user` 时**必填**）
  - `detected_at`: datetime | None（产生本命中记录的时间，建议 UTC）

**建议**：将 “paragraph” 视为 `span_type` 的一个取值，而不是单独再造一个 `SensitiveParagraphItem`，从而减少 schema 种类与链路复杂度。

#### 下一版本演进目标：RiskType Taxonomy（标准化语义层）

为避免 `risk_type` 命名漂移（phone/mobile/tel 等）导致统计与策略无法统一，建议引入标准 taxonomy：

- `risk_type`: 原始标签（规则/模型直接输出）
- `risk_type_normalized`: 标准标签（受控枚举或受控字符串）

示例 taxonomy（仅示意）：

- `PII_ID`
- `PII_PHONE`
- `FINANCIAL_ACCOUNT`
- `SECRET_KEY`
- `CONTRACT_RISK`

映射责任建议：

- 规则引擎命中：由规则侧直接输出 normalized
- LLM 输出：LLM 输出 raw，随后由规则表或映射函数标准化（或要求 LLM 直接输出 normalized）

---

### 5) `DocumentResultItem`（文档级汇总输出）

**用途**：对 Document 的最终结论、统计与可审计输出，供平台落库/前端展示。

- **必填**
  - `doc_id`: str
  - `archive_id`: str（若需要跨文档聚合）
  - `archive_owner_user_id`: str（与内嵌 span 一致，供校验）
  - `triggered_by_user_id`: str（与内嵌 span 一致，供校验）
  - `summary`: dict[str, Any]（总体风险分级、命中数量、关键证据摘要）
- **可选（成对出现以满足审计）**
  - `confirmed_by_user_id`: str（**谁确认**终局结论）
  - `confirmed_at`: datetime（**何时确认**；与 `confirmed_by_user_id` 须**同时有或同时无**）
- **可选**
  - `spans`: list[`SensitiveSpanItem`]（内嵌时注意 **文档体积**：海量 span 建议外置存储，仅保留引用或聚合指标）
  - `page_summaries`: list[dict]（每页统计/最强命中）
  - `artifacts`: list[`ArtifactRef`]（middle_json、渲染图等）

#### 下一版本演进目标：Span 外置化（避免大对象）

当命中数量可能达到 10k+ 时，建议将 spans 外置存储，并在结果中仅保留引用：

- `span_refs: ArtifactRef`（指向 spans JSON/Parquet 等）
- 或 `spans_embedded: bool` + `spans_ref: ArtifactRef`

默认建议：

- 生产路径优先外置化
- 内嵌仅用于小规模或 debug

---

## Pipeline 分段与算子 I/O 契约（A/B/C/D）

推荐将算子按“边界清晰的转换”切分，便于并行、重试与可观测。

```mermaid
flowchart LR
  ArchiveJobItem -->|A_ListDocuments| DocumentItem
  DocumentItem -->|B_ExtractPages| PageItem
  PageItem -->|C_DetectSensitive| SensitiveSpanItem
  SensitiveSpanItem -->|D_SummarizeDocument| DocumentResultItem
```

> 上图使用 **C2（span 流）**作为默认推荐（更利于索引/审计/聚合）；C1 方案见下节。

### A) `ArchiveJobItem -> DocumentItem*`（列文档）

- **职责**：展开 `source_paths`，生成每个 PDF 的 `DocumentItem`（补齐 `doc_id`、`mime_type`、可选 `file_hash`/`meta`）。
- **input_schema**：`OperatorSchema.item(ArchiveJobItem)`
- **output_schema**：`OperatorSchema.item(DocumentItem)`

失败语义建议：
- `source_paths` 为空或不可读：抛 `OperatorError(code="EMPTY_ARCHIVE"|"INVALID_SOURCE", retryable=False)`

### B) `DocumentItem -> PageItem*`（拆页/抽取）

- **职责**：解析 PDF 得到每页（`page_index` 0-based）并输出 `PageItem`。
- **input_schema**：`OperatorSchema.item(DocumentItem)`
- **output_schema**：`OperatorSchema.item(PageItem)`

并发建议：
- **document-level 解析**通常是 CPU/IO 混合，建议在 B 段内部控制并发（或在 pipeline 上按 doc 并行）。

### C) `PageItem -> SensitiveSpanItem*`（检测与定位）——默认推荐（C2）

- **职责**：对页文本/版面进行规则扫描或模型检测（含 LLM），输出 `SensitiveSpanItem` 流。
- **input_schema**：`OperatorSchema.item(PageItem)`
- **output_schema**：`OperatorSchema.item(SensitiveSpanItem)`

约束建议：
- 若产生 `bbox`，必须同时填充 `coord_space`，并保证与渲染/前端一致。
- `span_id` 必须稳定可复用（用于审计与高亮对齐）。

#### C 段：LLM 检测的审计与可复现建议（与 `docs/INFERENCE.md` 对齐）

当 C 段使用 LLM（OpenAI-compatible 网关）时，建议遵循：

- **Prompt 输入强约束**：采用 `PromptInputsModel` 校验运行时变量（缺字段 fail-fast），再渲染 prompt。
- **结构化输出**：优先 `response_format=json_schema` + Pydantic 校验，避免“半结构化文本”落地。
- **预算与截断**：若对 OCR 文本做 trimming，应在 LLM 审计中区分 original/effective（见下）。
- **缓存控成本**：允许算子显式开启 LLM cache（不同算子不同 TTL/Scope），并记录 cache_hit。

具体落地建议（写入 `SensitiveSpanItem.evidence`）：

- **llm_audit（推荐键）**：
  - `prompt_name` / `prompt_version`
  - `prompt_hash`（模板内容 hash）
  - `rendered_prompt_hash`（渲染后 hash；用于缓存与回放）
  - `output_schema_hash`（输出 JSON schema hash）
  - `request_fingerprint`（建议作为关联主键）
  - `model` / `base_url`
  - `is_truncated` + `original_input_chars` + `trimmed_input_chars`
  - `input_hash_original` / `input_hash_effective`
  - `cache_hit` / `cache_key_hash`（若启用缓存）
  - `finish_reason`（若为 length，应在推理层抛 TokenLimit 类错误而非重试）
  - `temperature` / `top_p` / `max_output_tokens`
  - `seed`（若网关/模型支持）
  - `execution_mode`: `deterministic` | `stochastic`
  - （可选）`request_snapshot_uri` / `response_snapshot_uri`（若启用 Payload Snapshot）

同时，`SensitiveSpanItem` 自身审计字段应与 LLM 调用结果保持一致：

- `detection_actor="system"`（自动）或 `"user"`（用户发起重跑/标注）
- `detected_by_user_id`：当 `detection_actor="user"` 时必填
- `detected_at`：命中生成时间（建议 UTC）

### D) `SensitiveSpanItem* -> DocumentResultItem`（文档级汇总）

> D 段通常需要知道 `doc_id` 的边界（span 流需要按 `doc_id` 分组）。如果你的 pipeline 执行模型是纯 “stream map”，建议 D 段由编排层完成（或在 D 段 operator 内做 group-by）。

- **职责**：聚合 spans、计算 doc 风险等级、输出 `DocumentResultItem`。
- **input_schema**：`OperatorSchema.item(SensitiveSpanItem)`
- **output_schema**：`OperatorSchema.item(DocumentResultItem)`

---

## Pipeline 成本控制：Predicate / Gating（建议）

为避免所有页面无条件进入 LLM 等高成本阶段，建议在编排层引入 gating（谓词）机制：

- 无文本（`text` 与 `text_artifact_ref` 均为空）→ 跳过 C 段或仅走低成本规则扫描
- OCR 质量过低（可写入 `page_meta["ocr_quality"]`）→ 降级或跳过
- 规则侧无命中且业务允许 → 早退出

实现形式可以是：

- 独立 Predicate Operator（filter/drop item）
- 或在 `page_meta` 中写入 `should_process_llm=false` 与 `skip_reason`

关键目标：显著降低 LLM 成本并提升吞吐。

---

## 错误语义标准化（建议）

`OperatorError(code=..., retryable=...)` 已具备基础能力。为提升可观测性与自动化重试策略，建议将 `code` 约束到受控集合（示例）：

- `INVALID_INPUT`
- `PARSE_FAILED`
- `OCR_FAILED`
- `LLM_TIMEOUT`
- `TOKEN_LIMIT`
- `SCHEMA_MISMATCH`
- `RATE_LIMIT`

并明确哪些 code 可重试（例如 RATE_LIMIT/LLM_TIMEOUT），哪些不可重试（INVALID_INPUT/TOKEN_LIMIT）。

---

## C1 vs C2：两种“敏感结果输出策略”与默认选择

### C1：`PageItem -> PageWithSensitiveSpans`（page 上挂 spans）

- **优点**
  - 以 page 为单位更贴近“人工复核 UI”的交互模型（右侧面板逐页展示命中）
  - 汇总时可直接按 page 聚合
- **缺点**
  - `PageWithSensitiveSpans` 会引入一个新 schema；在“硬匹配”机制下，下游必须明确声明接它而不是 `PageItem`
  - 对下游索引/检索来说，需要再做一次“拆 span”为独立事件

### C2：`PageItem -> SensitiveSpanItem*`（独立 span 流）——**默认推荐**

- **优点**
  - spans 天然是可审计事件（每条命中都有 provenance）
  - 更容易写入索引、做统计与跨 doc 分析
  - page 结构保持稳定（`PageItem` 作为主干 SSOT schema 不容易漂移）
- **缺点**
  - D 段需要显式分组（按 `doc_id`/`page_index`）来做 doc/page 汇总

---

## 如何在算子中声明 `input_schema()` / `output_schema()`（最小示例）

> 以下示例用于说明**声明方式与契约边界**，不展开具体 PDF 解析/OCR/模型调用逻辑。

```python
from platform_foundation.operators import BaseOperator, OperatorSchema, OperatorContext
from platform_foundation.contracts.pdf_items import ArchiveJobItem, DocumentItem


class PdfListDocumentsOperator(BaseOperator):
    op_name = "pdf_list_documents"
    op_version = "0.1.0"

    def input_schema(self) -> OperatorSchema:
        return OperatorSchema.item(ArchiveJobItem)

    def output_schema(self) -> OperatorSchema:
        return OperatorSchema.item(DocumentItem)

    def process_item(self, ctx: OperatorContext, item: dict) -> dict:
        ...
```

> 约定：**schema 模型只在 `platform_foundation/contracts/` 定义**，算子只 import 引用，避免“每个算子各自定义一份 PageItem”导致链路不兼容。

---

## 与现有文件夹结构的适配建议

- **SSOT schemas（新增）**：`backend/foundation/platform_foundation/contracts/`  
  - `pdf_items.py`：本文定义的 `ArchiveJobItem/DocumentItem/PageItem/SensitiveSpanItem/DocumentResultItem`  
  - （可选）`provenance.py`：将 `00_cross_module_contracts.md` §5 的最小集固化为模型（供 spans/evidence 复用）
- **算子实现（新增/扩展）**：`backend/foundation/platform_foundation/operators/`  
  - `pdf/` 子目录（可选）：按 A/B/C/D 分段组织，如 `operators/pdf/list_documents.py`、`operators/pdf/extract_pages.py` 等
- **适配层（如需要 Ray/Dataset）**：`backend/foundation/platform_foundation/operators/adapters/`（已存在 `ray_data.py`/`daft.py`）

---

## 验收清单（写完/实现前自检）

- **契约完整**：每段算子都有明确的 `input_schema()` / `output_schema()`，并引用 SSOT 模型。
- **ID 可回溯**：`doc_id/page_index/span_id` 足够支撑审计与 UI 高亮回放。
- **几何一致**：若输出 `bbox`，必须同时输出 `coord_space` 且与渲染链路一致。
- **溯源最小集**：敏感命中至少具备 provenance 最小字段（`source/page_index/char_start/char_end`…）。
- **默认策略明确**：C2（span 流）作为默认推荐；若选 C1，明确下游 schema 与汇总方式。


