# Requirement 模块设计文档（并行开发）

本目录把 [highlevel_direction.md](../highlevel_direction.md) 中的技术主线拆成**可并行**的设计说明；与仓库权威设计 **[docs/design/](../../design/)**、**[docs/API_CONTRACT.md](../../API_CONTRACT.md)** 配合阅读，避免重复粘贴大段正文。

---

## ⚠️ 重要更新（2026-04）

### 新增核心文档

| 文档 | 说明 |
|------|------|
| [`docs/OPERATORS.md`](../../OPERATORS.md) | 算子库完整参考（从代码 README 迁移并扩展） |
| [`docs/PIPELINE.md`](../../PIPELINE.md) | 流水线编排完整参考（新建） |

### 架构演进：Foundation / Services 拆分

系统正在向**模块化架构**演进，详见：
- **重构计划**：[backend_modular_refactor_6b64915a.plan.md](../../../.cursor/plans/backend_modular_refactor_6b64915a.plan.md)
- **架构愿景**：[../modularization_vision.md](../modularization_vision.md)

**核心变化**：
- `app/operators/` + `app/pipeline/` → `foundation/` 包（可独立发布）
- 各业务域（doc-service、extraction-service 等）→ `services/` 目录
- `foundation` 保持**无 FastAPI/SQLAlchemy**，通过 Protocol 注入依赖

---

## 与 ownership 的对应关系

| 设计文档 | highlevel 条目 | 主要负责人（见 [ownership.md](../ownership.md)） |
|----------|----------------|--------------------------------------------------|
| [00_cross_module_contracts.md](00_cross_module_contracts.md) | 横切契约 | 总负责牵头，各模块评审 |
| [01_ocr_layout_mineru.md](01_ocr_layout_mineru.md) | 1 OCR 与结构化解析 | 邵海洋 |
| [02_text_alignment_provenance.md](02_text_alignment_provenance.md) | 2 文本定位与对齐 | 邵海洋、陈佳立 |
| [03_metadata_extraction.md](03_metadata_extraction.md) | 3 元数据抽取 | 陈佳立 |
| [04_catalog_rule_engine.md](04_catalog_rule_engine.md) | 4 编目规则引擎 | 蒲睿韬 |
| [05_platform_pipeline_ray_operators.md](05_platform_pipeline_ray_operators.md) | 5 分布式 Pipeline、6 算子标准化 | **总负责（建议）**；highlevel 第 5、6 条在 ownership 表中未单列，由本文档统一覆盖 |
| [06_llm_multi_agent_review.md](06_llm_multi_agent_review.md) | 7 LLM + Multi-Agent 审核 | 徐仕林 |
| [07_sensitive_detection.md](07_sensitive_detection.md) | 8 敏感信息识别 | 蒲睿韬 |
| [08_retrieval_rag.md](08_retrieval_rag.md) | 9 检索与 RAG | 奉仰麟 |
| [09_data_agent_nl2query.md](09_data_agent_nl2query.md) | 10 Data Agent | 徐仕林 |

---

## 推荐阅读顺序

### 新成员快速上手
1. [docs/README.md](../../README.md) - 项目文档总览
2. [docs/design/01_system_goal_and_architecture.md](../../design/01_system_goal_and_architecture.md) - 系统目标与架构（10分钟）
3. [docs/OPERATORS.md](../../OPERATORS.md) - 算子库（5分钟）
4. [docs/PIPELINE.md](../../PIPELINE.md) - 流水线编排（5分钟）
5. 本目录 `00_cross_module_contracts.md` - 跨模块契约（10分钟）
6. 与自己模块相关的 `01`–`09`

### 深度阅读路径

| 角色 | 阅读路径 |
|------|----------|
| 全栈开发 | `01` → `02` → `03` → `06` → `07` |
| AI/ML 方向 | `01` → `02` → `06` → `07` → `08` |
| 平台/架构方向 | `00` → `05` → `06` → [modularization_vision.md](../modularization_vision.md) |
| 前端方向 | [docs/design/03_frontend_module_design.md](../../design/03_frontend_module_design.md) |

---

## 权威设计文档索引（勿在本目录重复展开）

| 主题 | 路径 |
|------|------|
| 系统目标与状态机 | [docs/design/01_system_goal_and_architecture.md](../../design/01_system_goal_and_architecture.md) |
| 后端模块 | [docs/design/02_backend_module_design.md](../../design/02_backend_module_design.md) |
| 前端模块 | [docs/design/03_frontend_module_design.md](../../design/03_frontend_module_design.md) |
| AI Provider 与 Step0–3 | [docs/design/04_ai_pipeline_design.md](../../design/04_ai_pipeline_design.md) |
| 人工复核与审计 | [docs/design/05_human_override_audit_state_machine.md](../../design/05_human_override_audit_state_machine.md) |
| Platform Jobs、Ray、Rulesets | [docs/design/06_document_intelligence_platform.md](../../design/06_document_intelligence_platform.md) |
| 批量与共享存储 | [docs/design/AI批量异步处理方案-共享存储版.md](../../design/AI批量异步处理方案-共享存储版.md) |

---

## 单篇文档统一结构（模板）

各 `0x_*.md` 建议包含以下小节（便于不同水平读者自学）：

1. **文档目的与读者**（初级需先读哪些；中级可从「契约」节开始）
2. **实现状态**（✅ 已实现 / 🔄 进行中 / 📋 规划）
3. **与需求条目的映射**（对应 `highlevel_direction.md` 编号）
4. **现状：已实现能力与代码地图**（表格：能力 | 代码路径 | 权威设计链接）
5. **与平台可共用能力的差距**（配置快照、租户隔离、可观测性、幂等、可替换 Provider 等）
6. **目标形态与接口**（本模块对外承诺的字段/事件；细节引用 `00`）
7. **与其他模块的并行约定**（依赖谁 / 谁依赖我 / 可 mock 数据）
8. **代码阅读建议**（5 分钟 / 30 分钟 / 半天 三条路径）
9. **思考题与自测**（2–4 题）

---

## 并行开发协作要点

- 模块间只通过 **[00_cross_module_contracts.md](00_cross_module_contracts.md)** 中约定的 **IDL** 耦合；接口变更先改 `00` 再改实现
- 后端实现以 `app/pipeline` → `app/operators` 单向依赖为准（见 [docs/OPERATORS.md](../../OPERATORS.md)）
- **架构演进**：未来将拆分为 `foundation/`（纯 Python 库）+ `services/`（业务服务），见 [../modularization_vision.md](../modularization_vision.md)
- API 变更必须同步 **[docs/API_CONTRACT.md](../../API_CONTRACT.md)**（本目录仅做需求级设计，不替代契约）
