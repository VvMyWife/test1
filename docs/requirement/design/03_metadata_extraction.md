# 设计：元数据抽取（规则 + RAG + LLM）

**对应需求**：[highlevel_direction.md](../highlevel_direction.md) 第 3 条。  
**主要负责人**：陈佳立（见 [ownership.md](../ownership.md)）。  
**前置阅读**：[00_cross_module_contracts.md](00_cross_module_contracts.md) §5–§7；[04_catalog_rule_engine.md](04_catalog_rule_engine.md)（规则与编目边界）。

---

## 1. 文档目的与读者

- **初级**：分清「CSV 规则扫描」「YAML 关键词」「LLM Step1–3」三条路径各自入口。
- **中级**：设计「抽取结果 + 置信度 + fallback」统一 schema，并与 `ruleset` / `config_snapshot` 对齐。

---

## 2. 现状：已实现能力与代码地图


| 能力                             | 说明                                        | 代码 / 文档路径                                                                                                                                                                                           |
| ------------------------------ | ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CSV 规则扫描（SSOT 声明在 pipeline 日志） | `scan` / `scan_with_rules` → `list[dict]` | [backend/app/rules_engine/**init**.py](../../../backend/app/rules_engine/__init__.py)、[backend/app/rules_engine/scan_v1.py](../../../backend/app/rules_engine/scan_v1.py)                           |
| YAML 关键词引擎                     | `KeywordScanEngine`，与 CSV 为**两条路径**       | [backend/app/inference/keyword_scan.py](../../../backend/app/inference/keyword_scan.py)；合并见 [backend/app/operators/static_lexicon_merge.py](../../../backend/app/operators/static_lexicon_merge.py) |
| LLM 流水线 Step1–3                | 路由、子 Agent、裁决                             | [backend/app/inference/pipeline.py](../../../backend/app/inference/pipeline.py)；设计 **[docs/design/04_ai_pipeline_design.md](../../design/04_ai_pipeline_design.md)**                                |
| 配置加载                           | `ai_analysis.yaml`、模型配置                   | [backend/app/inference/config_loader.py](../../../backend/app/inference/config_loader.py)                                                                                                           |
| 图片流水线内规则命中                     | OCR 文本 → `scan_sensitive_rules`           | [backend/app/operators/scan_rules_hits.py](../../../backend/app/operators/scan_rules_hits.py)、`bridge_rules`                                                                                        |
| 平台配置快照                         | 任务创建时冻结 `resolved_config`                 | **[docs/design/06_document_intelligence_platform.md](../../design/06_document_intelligence_platform.md)**、`config_snapshot`                                                                         |


**RAG（向量检索 + 知识库辅助抽取）**：当前仓库**未发现**独立的 ES/向量服务集成与 embedding 流水线；预研与落地见 [08_retrieval_rag.md](08_retrieval_rag.md)。本模块文档将 RAG 视为**规划能力**，不在此假装已实现。

---

## 3. 与平台可共用能力的差距（优化方向）

1. **策略分层**：明确「规则优先 / LLM 优先 / 混合」与 [00 `rule_ai_merge_policy](00_cross_module_contracts.md)` 一致；避免 Service 层多处硬编码。
2. **统一抽取 schema**：建议所有结构化字段携带 `value`、`confidence`、`source`（`regex`/`llm`/`rag`）、`provenance`（见 [00 §5](00_cross_module_contracts.md)），便于人工复核与审计。
3. **不确定性**：低置信度时的 fallback（重试、换模型、标为「待人工」）需要与 **文档状态机** 对齐（**[docs/design/01_system_goal_and_architecture.md](../../design/01_system_goal_and_architecture.md)**）。
4. **租户与版本**：`ruleset_version` + `config_snapshot` 应能复现一次抽取；禁止执行路径上「静默」拉取最新 ruleset。
5. **双路径词库**：CSV 与 YAML 关键词并存（见 `rules_engine` 模块注释）；长期应合并或明确优先级文档（可参考 `docs/DEPLOYMENT/CONFIG_SOURCES.md`，若仓库内有）。

---

## 4. 目标形态与接口（本模块对外承诺）

- **输入**：规范化全文或分页文本；可选 `rules_context`；平台任务携带 `config_snapshot`。
- **输出**：结构化键值 + 置信度 + 证据引用；写库经 **Service 层事务**（**AGENT_RULES / CLAUDE.md**）。
- **不承诺**：在本模块内直接改 `Document.status`；状态推进走既有服务。

---

## 5. 与其他模块的并行约定


| 关系      | 说明                                                                                                                            |
| ------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **我依赖** | [02](02_text_alignment_provenance.md)（字段级 evidence）；[04](04_catalog_rule_engine.md)（编目规则与分类体系）                                |
| **我提供** | 元数据字段、置信度、溯源，供审核与检索消费                                                                                                         |
| **依赖我** | [06](06_llm_multi_agent_review.md)（若抽取结果作为 Agent 输入）；[08](08_retrieval_rag.md)（索引字段来源）；[09](09_data_agent_nl2query.md)（语义层指标） |


**Mock**：对 `invoke_llm_json` 打桩；规则侧用 `scan_with_rules` 固定短文本。

---

## 6. 代码阅读建议


| 时长     | 路径                                                                                             |
| ------ | ---------------------------------------------------------------------------------------------- |
| ~5 分钟  | `rules_engine.scan`、`inference/pipeline.run_pipeline` 函数签名与返回值键名                               |
| ~30 分钟 | `run_step1` / `run_step2` / `run_step3`；`build_rules_pipeline_context`（`rules_loader`）         |
| ~半天    | `config_loader` 与 ruleset zip 解析（若走平台 ruleset）；对照 **04_ai_pipeline_design** 与 **API_CONTRACT** |


---

## 7. 思考题与自测

1. `pipeline.py` 日志称 Step0 SSOT 为 CSV — 若启用 YAML `KeywordScanEngine`，最终提示词里关键词以谁为准？
2. 同一字段规则命中与 LLM 结果冲突时，你希望 DB 里保留一条记录还是两条？如何审计？
3. 引入 RAG 后，`provenance` 应指向「检索 chunk」还是「原文 span」？能否同时指向二者？

