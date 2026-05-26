# 设计：编目规则引擎（Rule Engine）

**对应需求**：[highlevel_direction.md](../highlevel_direction.md) 第 4 条。  
**主要负责人**：蒲睿韬（见 [ownership.md](../ownership.md)）。  
**前置阅读**：[00_cross_module_contracts.md](00_cross_module_contracts.md) §6–§7；[03_metadata_extraction.md](03_metadata_extraction.md)。

---

## 1. 文档目的与读者

- **初级**：理解当前「敏感/开放审查规则」与「档案编目字段规则」在代码中**尚未同构**；先掌握 `scan_with_rules` 契约。
- **中级**：起草 JSON/YAML DSL、多分类体系、与 AI 结果的 override/校验顺序。

---

## 2. 现状：已实现能力与代码地图


| 能力             | 说明                                                  | 代码 / 文档路径                                                                                                                                                       |
| -------------- | --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CSV 规则扫描 v1    | 文本 → 命中列表（字典字段与 CSV 列相关）                            | [backend/app/rules_engine/scan_v1.py](../../../backend/app/rules_engine/scan_v1.py)                                                                             |
| 契约测试           | 不依赖内部实现细节                                           | [backend/tests/rules_engine/test_scan_contract.py](../../../backend/tests/rules_engine/test_scan_contract.py)                                                   |
| 与算子 bridge     | `scan_sensitive_rules`                              | [backend/app/operators/support/bridge_rules.py](../../../backend/app/operators/support/bridge_rules.py)                                                         |
| 规则 + Agent 上下文 | `build_rules_pipeline_context`、`get_rules_by_agent` | [backend/app/inference/rules_loader.py](../../../backend/app/inference/rules_loader.py)                                                                         |
| 平台 Ruleset 制品  | zip/文件上传、版本、`published`                             | **[docs/design/06_document_intelligence_platform.md](../../design/06_document_intelligence_platform.md)**；**[docs/API_CONTRACT.md](../../API_CONTRACT.md)** §13 |


**编目（档案馆业务字段、多级分类、档号规则等）**：当前 **rules_engine** 主要服务**敏感与审查流水线**；「编目逻辑完全配置化」属于本设计文档要推动的**目标**，需新增或扩展规则类型与执行引擎，与 CSV 扫描**解耦演进**（避免一次性推翻 SSOT）。

---

## 3. 与平台可共用能力的差距（优化方向）

1. **DSL 与存储**：从「代码 + CSV」演进到 **JSON/YAML DSL**（条件、动作、优先级、分类体系 ID）；制品形态与 **ruleset 版本**一致，便于租户级发布回滚。
2. **多分类体系**：支持多标准并行（单位 A / 单位 B），输出带 `scheme_id`；与 **metadata 抽取**字段对齐。
3. **规则与 AI 融合**：在 [00](00_cross_module_contracts.md) 选定 `rule_ai_merge_policy`；编目场景常见「规则校验 LLM 建议的档号」。
4. **执行顺序**：优先级、短路、DAG 子集；与平台 **DAG-lite**（**06 设计文档** §5）对齐，避免 Celery/Ray 阶段与规则阶段顺序漂移。
5. **测试与金样**：像 `test_scan_contract.py` 一样，为编目 DSL 增加 **golden cases**（输入文档特征 → 期望编目字段）。

---

## 4. 目标形态与接口（本模块对外承诺）

- **输入**：结构化文档特征（元数据、全文、可选版面块）；租户 + `ruleset_version_id` 或内联 DSL 版本。
- **输出**：编目字段补丁 + 命中规则 ID 列表 + 解释信息；冲突时显式 `conflicts[]`。
- **与现有 scan 关系**：短期可并存：敏感扫描继续走 `scan_v1`；编目走新引擎或扩展表结构——**接口上**通过不同 `rule_profile` 区分。

---

## 5. 与其他模块的并行约定


| 关系      | 说明                                                                                                                           |
| ------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **我依赖** | [01](01_ocr_layout_mineru.md)/[02](02_text_alignment_provenance.md) 若规则需版面块；[03](03_metadata_extraction.md) 的 LLM 抽取结果作为输入之一 |
| **我提供** | 编目决策、校验错误、与敏感/开放规则可组合的优先级定义                                                                                                  |
| **依赖我** | [06](06_llm_multi_agent_review.md)（裁决是否采纳某编目建议）；平台任务 `resolved_config` 中的规则段                                                 |


**Mock**：纯函数规则集 JSON + 小型 fixture 文档特征 dict。

---

## 6. 代码阅读建议


| 时长     | 路径                                                                    |
| ------ | --------------------------------------------------------------------- |
| ~5 分钟  | `rules_engine/__init__.py` 注释（CSV vs YAML 路径）；`test_scan_contract.py` |
| ~30 分钟 | `scan_v1.py`；`rules_loader.build_rules_pipeline_context`              |
| ~半天    | 平台 ruleset 上传与解析路径（`api/v1/rulesets` 相关 service/model）；**06 设计文档** 全文 |


---

## 7. 思考题与自测

1. 编目规则「条件」应基于 OCR 全文还是结构化字段？误判代价哪种更大？
2. `published` ruleset 不可变 — 编目规则发现错误时，如何在不破坏旧任务的前提下修复？
3. 多分类体系下，同一文档两个互斥分类同时命中，DSL 应表达「禁止」还是交给 Step3 仲裁？

