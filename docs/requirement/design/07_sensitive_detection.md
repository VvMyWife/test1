# 设计：敏感信息识别（规则 + 模型融合）

**对应需求**：[highlevel_direction.md](../highlevel_direction.md) 第 8 条。  
**主要负责人**：蒲睿韬（见 [ownership.md](../ownership.md)）。  
**前置阅读**：[00_cross_module_contracts.md](00_cross_module_contracts.md) §6；[02_text_alignment_provenance.md](02_text_alignment_provenance.md)；[06_llm_multi_agent_review.md](06_llm_multi_agent_review.md)。

---

## 1. 文档目的与读者

- **初级**：区分「图片流水线里的规则扫描算子」与「版面几何定位算子」；会读 `SensitiveHit`。
- **中级**：设计 NER/LLM 与 regex 的优先级、敏感分类体系、precision 优先的评测。

---

## 2. 现状：已实现能力与代码地图


| 能力        | 说明                                   | 代码 / 文档路径                                                                                                                   |
| --------- | ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| 规则扫描算子    | OCR 清洗文本 → hits                      | [backend/app/operators/scan_rules_hits.py](../../../backend/app/operators/scan_rules_hits.py) `SensitiveDetectOperator`     |
| 状态模型      | `ImagePipelineState.hits`            | [backend/app/operators/image_pipeline_models.py](../../../backend/app/operators/image_pipeline_models.py)                   |
| 几何定位      | `middle.json` + hits → `GeometryHit` | [backend/app/operators/sensitive_geometry_from_layout.py](../../../backend/app/operators/sensitive_geometry_from_layout.py) |
| 系统标注同步    | 写入 `document_annotations`            | [backend/app/services/annotation_service.py](../../../backend/app/services/annotation_service.py)                           |
| CSV SSOT  | `scan_with_rules`                    | [backend/app/rules_engine/](../../../backend/app/rules_engine/)                                                             |
| LLM 侧敏感结论 | Step2 各 Agent、`is_sensitive` 等       | [backend/app/inference/pipeline.py](../../../backend/app/inference/pipeline.py)                                             |


**NER 专用管线**：当前仓库未看到独立「NER 模型服务」模块；融合策略在本文件中以**规划**形式给出，落地时新增 provider 或算子。

---

## 3. 与平台可共用能力的差距（优化方向）

1. **统一敏感分类体系**：regex 类别、LLM `sensitivity_tags`、DB `label` 使用同一套枚举或映射表（可租户扩展）。
2. **多策略优先级**：建议默认 **precision 优先**（强规则 > NER > LLM），弱信号仅作提示；与 [06](06_llm_multi_agent_review.md) 裁决衔接。
3. **误判治理**：灰度发布词库、线上误报反馈回路、金样集回归（与 `test_scan_contract` 类似）。
4. **可观测性**：按 `hit_origin` 统计命中率；平台 `metrics` 可区分 `regex` vs `llm`。
5. **隐私**：敏感片段落库脱敏策略（hash、截断）与审计要求（**05_human_override**）。

---

## 4. 目标形态与接口（本模块对外承诺）

- **输入**：全文或分页文本；可选 `middle.json` 用于几何；租户规则配置。
- **输出**：`SensitiveHit` / `GeometryHit` / 标注记录；与 [00 §6](00_cross_module_contracts.md) 字段兼容。
- **融合**：输出中显式 `hit_origin`，禁止静默覆盖强规则结果（除非走仲裁策略）。

---

## 5. 与其他模块的并行约定


| 关系      | 说明                                                                                                                      |
| ------- | ----------------------------------------------------------------------------------------------------------------------- |
| **我依赖** | [01](01_ocr_layout_mineru.md)、[02](02_text_alignment_provenance.md)（几何）；[04](04_catalog_rule_engine.md)（若敏感词纳入 ruleset） |
| **我提供** | hits、标注、供 Step1/Step2 的候选证据                                                                                             |
| **依赖我** | [06](06_llm_multi_agent_review.md)（Step0 输入）；前端高亮（**03_frontend**）                                                      |


**Mock**：固定 `ocr_text_clean` + 小 CSV fixture；几何用最小 `middle.json`。

---

## 6. 代码阅读建议


| 时长     | 路径                                                                                         |
| ------ | ------------------------------------------------------------------------------------------ |
| ~5 分钟  | `SensitiveDetectOperator`、`SensitiveHit` 定义                                                |
| ~30 分钟 | `sensitive_geometry_from_layout`、`annotation_service.sync_system_annotations_for_document` |
| ~半天    | `rules_engine/scan_v1`；`pipeline.run_step2` 中单 Agent prompt 构造                             |


---

## 7. 思考题与自测

1. 仅有 regex 命中而无 LLM 确认时，开放结论应偏保守还是激进？与业务「延期开放」默认值是否一致？
2. NER 与 regex 在同一 span 冲突时，UI 上显示几个框？
3. 如何将「敏感类型」映射到国标/行标，以满足多单位扩展（见 [04](04_catalog_rule_engine.md)）？

