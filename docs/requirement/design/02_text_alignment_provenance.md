# 设计：文本定位与对齐（溯源基础能力）

**对应需求**：[highlevel_direction.md](../highlevel_direction.md) 第 2 条。  
**主要负责人**：邵海洋、陈佳立（见 [ownership.md](../ownership.md)）。  
**前置阅读**：[00_cross_module_contracts.md](00_cross_module_contracts.md) §3–§5；[01_ocr_layout_mineru.md](01_ocr_layout_mineru.md)。

---

## 1. 文档目的与读者

- **初级**：理解「全文字符串」与「页上框」如何通过 `middle.json` 关联；会读 `DocumentAnnotation` 字段。
- **中级**：设计跨模块的 `provenance` schema 与前端高亮 API 对齐方案。

---

## 2. 现状：已实现能力与代码地图

| 能力 | 说明 | 代码 / 文档路径 |
|------|------|-----------------|
| Span 迭代 | `page_idx`、`page_size`、`bbox`、`content` | [backend/app/services/mineru_parser.py](../../../backend/app/services/mineru_parser.py) `iter_spans_from_middle` |
| 页内 offset → bbox | 按 span 顺序拼接全文，记录 `[char_start,char_end)→bbox` | [backend/app/operators/sensitive_geometry_from_layout.py](../../../backend/app/operators/sensitive_geometry_from_layout.py) `_build_page_text_and_ranges`、`_bboxes_for_range` |
| 几何命中 DTO | `GeometryHit`（`phrase`、`page_index`、`bboxes`、`coord_space`） | [backend/app/operators/support/document_contracts.py](../../../backend/app/operators/support/document_contracts.py) |
| 持久化标注 | `bbox` JSONB、`coord_space`、`page_index`、`matched_text` | [backend/app/models/document_annotation.py](../../../backend/app/models/document_annotation.py)；迁移 `0012_document_annotations` |
| 同步系统标注 | AI 结果 → 标注 | [backend/app/services/annotation_service.py](../../../backend/app/services/annotation_service.py) |
| PDF 叠加辅助检索 | span 写入隐形文字层 | [backend/app/services/pdf_overlay.py](../../../backend/app/services/pdf_overlay.py) `generate_dual_layer_pdf` |

**前端与交互**：三栏布局与审阅展示见 **[docs/design/03_frontend_module_design.md](../../design/03_frontend_module_design.md)**（具体高亮组件以实现为准）。

**注意**：DB 默认 `coord_space` 历史上为 `mineru_pdf`，代码常量 `MINERU_LAYOUT_COORD_SPACE = "mineru_layout"`；集成时须核对前后端与 overlay 使用同一解释（见 [00](00_cross_module_contracts.md) §3）。

---

## 3. 与平台可共用能力的差距（优化方向）

1. **统一 provenance SSOT**：当前敏感路径有页内 range + bbox；**元数据字段、LLM evidence、RAG chunk** 应复用 [00 §5](00_cross_module_contracts.md) 最小字段集，并**锁定**字符编码偏移（UTF-8 vs UTF-16）。
2. **稳定 `span_id`**：若 `middle.json` 无全局 ID，需在 pipeline 生成稳定 ID（如 content hash + 页 + 序号），便于审计引用与去重。
3. **跨页引用**：长实体跨页时，需约定多 `provenance[]` 还是单记录多 bbox；影响前端高亮与导出。
4. **AI 输出回溯**：`AIResult` 内 JSON 字段与 `provenance` 绑定方式尚未统一；见 [06_llm_multi_agent_review.md](06_llm_multi_agent_review.md)。
5. **API 暴露**：列表/详情是否返回标注与坐标，以 **[docs/API_CONTRACT.md](../../API_CONTRACT.md)** 当前文档为准；扩展时保持统一响应结构（`success`/`data`/`error`）。

---

## 4. 目标形态与接口（本模块对外承诺）

- **输入**：`middle.json` 路径；可选「全文上的子串或页内 offset」。
- **输出**：`GeometryHit[]` 或更广义的 `ProvenanceRecord[]`（在实现中扩展但兼容 [00 §5](00_cross_module_contracts.md)）。
- **坐标系**：每条几何记录必须带 `coord_space`，与 [01](01_ocr_layout_mineru.md) 输出一致。

---

## 5. 与其他模块的并行约定

| 关系 | 说明 |
|------|------|
| **我依赖** | [01](01_ocr_layout_mineru.md) 的 `middle.json` 与坐标系 |
| **我提供** | 页内对齐、bbox 列表、可持久化标注数据 |
| **依赖我** | [03](03_metadata_extraction.md)（字段级 evidence）、[06](06_llm_multi_agent_review.md)、[07](07_sensitive_detection.md)、[08](08_retrieval_rag.md) |

**Mock**：使用仓库内小型 `middle.json` fixture + 已知短语，验证 range→bbox 映射。

---

## 6. 代码阅读建议

| 时长 | 路径 |
|------|------|
| ~5 分钟 | `iter_spans_from_middle` 数据结构；`GeometryHit` 字段 |
| ~30 分钟 | `sensitive_geometry_from_layout` 全文拼接与 `_bboxes_for_range` |
| ~半天 | `annotation_service.sync_system_annotations_for_document`；`pdf_overlay` 坐标变换 `_scale_mineru_bbox_to_page_rect`；对照前端审阅区 |

---

## 7. 思考题与自测

1. 同一页两个 span 拼接无分隔符时，子串匹配歧义如何避免？是否需要在拼接处插入零宽标记（仅内部索引）？
2. 用户在前端修改了 OCR 文本后，原有 bbox 是否全部失效？需要何种「修订层」模型？
3. `DocumentAnnotation.source` 区分 system/user 的意义是什么？复核流程见 **[docs/design/05_human_override_audit_state_machine.md](../../design/05_human_override_audit_state_machine.md)** — 试举一条状态变更与标注关系。
