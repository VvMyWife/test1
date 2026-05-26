# 设计：检索与 RAG（结构化 + 全文 + 向量）

**对应需求**：[highlevel_direction.md](../highlevel_direction.md) 第 9 条。  
**主要负责人**：奉仰麟（见 [ownership.md](../ownership.md)）。  
**前置阅读**：[00_cross_module_contracts.md](00_cross_module_contracts.md) §5；[02_text_alignment_provenance.md](02_text_alignment_provenance.md)；[03_metadata_extraction.md](03_metadata_extraction.md)。

---

## 1. 文档目的与读者

- **初级**：理解当前「检索」主要靠 DB API +（可选）PDF 隐形文字层，**无独立 ES/向量服务**。
- **中级**：设计三类检索（metadata / 全文 / 向量）的租户隔离、chunk 策略与和编目纠错的闭环。

---

## 2. 现状：已实现能力与代码地图

| 能力 | 说明 | 代码 / 文档路径 |
|------|------|-----------------|
| 结构化查询（文档列表/详情） | `database_id`、`status`、`risk_level` 等过滤 | **[docs/API_CONTRACT.md](../../API_CONTRACT.md)** §4.2、§4.3；实现见 `api/v1/documents` |
| 档案层级浏览 | 户/卷/件/页 | **API_CONTRACT** §3.4 |
| PDF 隐形文字层（辅助检索） | MinerU span → `insert_text` render_mode=3 | [backend/app/services/pdf_overlay.py](../../../backend/app/services/pdf_overlay.py) `generate_dual_layer_pdf` |
| 元数据与 AI 结果 | 随文档与 `AIResult` 存储 | **[docs/DATABASE_SCHEMA.md](../../DATABASE_SCHEMA.md)**（若需表字段） |

**Elasticsearch / OpenSearch**：仓库内 **未发现** 相关依赖与客户端代码（全库检索 `elasticsearch`/`opensearch` 无匹配）。

**向量检索 / Embedding / RAG 流水线**：**未发现** 独立向量库或 embedding 任务；与 [03](03_metadata_extraction.md) 所述一致，RAG 为**规划能力**。

---

## 3. 与平台可共用能力的差距（优化方向）

1. **三类检索统一网关**：对外抽象 `SearchRequest`（过滤 + 可选全文 query + 可选 vector query），内部路由到 PG/ES/向量库；避免前端直连多个系统。
2. **Chunk 策略**：与 [02](02_text_alignment_provenance.md) 页/段/span 一致；每条 chunk 携带 `provenance` 以便 RAG 回答可定位。
3. **与元数据抽取打通**：抽取字段进入 **可检索索引**（含同义词/归一化）；编目纠错可触发「检索相似已发布档案」。
4. **租户隔离**：索引名前缀或强制 `tenant_id` 过滤；与 **API_CONTRACT** §2.5 平台租户头一致。
5. **Embedding 选型**：预研输出建议包含：模型、维度、语言、成本、是否本地部署；与 **05** GPU/队列策略一起评审。

---

## 4. 目标形态与接口（本模块对外承诺）

- **阶段 1（可落地）**：扩展 **API_CONTRACT** 文档与实现：在现有 PG 上支持常用 metadata 组合查询、排序、导出；全文检索可先依赖 DB `LIKE`/pg_trgm 或引入 ES（二选一写 ADR）。
- **阶段 2**：引入向量索引 + RAG 服务；API 与 **统一响应结构**（**API_CONTRACT** §2.4）一致。

本文件不虚构具体路径；落地后应在 **API_CONTRACT** 增加「检索/RAG」章节。

---

## 5. 与其他模块的并行约定

| 关系 | 说明 |
|------|------|
| **我依赖** | [02](02_text_alignment_provenance.md) chunk 与溯源；[03](03_metadata_extraction.md) 字段；[01](01_ocr_layout_mineru.md) 版面结构 |
| **我提供** | 检索结果、RAG 引用片段 → 供编目/审核辅助 |
| **依赖我** | [09_data_agent_nl2query.md](09_data_agent_nl2query.md)（Tool 调用检索） |

**Mock**：内存 list[dict] 文档 fixture；向量阶段用固定 embedding 假数据。

---

## 6. 代码阅读建议

| 时长 | 路径 |
|------|------|
| ~5 分钟 | **API_CONTRACT** §4.2 查询参数；`pdf_overlay.generate_dual_layer_pdf` 注释 |
| ~30 分钟 | `documents` 相关 router + service；`DATABASE_SCHEMA` 文档表 |
| ~半天 | 设计 PG 索引 vs 引入 ES 的 POC（本仓库外可单独分支） |

---

## 7. 思考题与自测

1. 隐形 PDF 层检索依赖桌面端查看器能力；Web 端全文检索是否仍需 ES？
2. RAG 回答引用 chunk 时，如何防止跨租户数据通过 embedding 近似检索泄漏？
3. 「编目纠错」场景下，检索应优先相似 **元数据** 还是相似 **正文**？
