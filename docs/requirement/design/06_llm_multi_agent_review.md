# 设计：LLM + Multi-Agent 审核系统

**对应需求**：[highlevel_direction.md](../highlevel_direction.md) 第 7 条。  
**主要负责人**：徐仕林（见 [ownership.md](../ownership.md)）。  
**前置阅读**：[00_cross_module_contracts.md](00_cross_module_contracts.md) §5、§7；**[docs/design/04_ai_pipeline_design.md](../../design/04_ai_pipeline_design.md)**（权威）。

---

## 1. 文档目的与读者

- **初级**：跟随 Step0→Step1→Step2→Step3 数据流，能指出每步输入输出键名。
- **中级**：扩展 `evidence`/`reason` 结构、Agent 插件化与租户级 prompt 版本。

---

## 2. 现状：已实现能力与代码地图

| 能力 | 说明 | 代码 / 文档路径 |
|------|------|-----------------|
| Step1 路由 | 摘要、`selected_agent_ids` | [backend/app/inference/pipeline.py](../../../backend/app/inference/pipeline.py) `run_step1` |
| Step2 子 Agent | 线程池并发 `_run_single_agent` | 同上 `run_step2` |
| Step3 裁决 | `open_decision`、风险、标签、`reasoning` 等 | 同上 `run_step3` |
| 整条流水线 | `run_pipeline` | 同上 `run_pipeline` |
| LLM 调用 | JSON 模式 | [backend/app/inference/providers/llm_client.py](../../../backend/app/inference/providers/llm_client.py) `invoke_llm_json` |
| 规则上下文 | Agent、规则按 ID 组织 | [backend/app/inference/rules_loader.py](../../../backend/app/inference/rules_loader.py) |

**架构原则**（与设计文档一致）：Provider 不写库；**Service** 负责事务、状态、`AIResult` 追加（**04_ai_pipeline_design** §3）。

---

## 3. 与平台可共用能力的差距（优化方向）

1. **可解释性字段**：在 LLM JSON 中强制 **`evidence`（引用原文片段或 span_id）** 与 **`reason`（短理由）**，并与 [02](02_text_alignment_provenance.md) 的 `provenance` 绑定，避免仅有一句 `reasoning` 无法定位。
2. **Agent 插件化**：当前子 Agent 由 CSV/`rules_loader` 驱动；平台化需支持 **ruleset 内声明 Agent** + 模型配置快照。
3. **协作模式**：现为「路由 + 并发 + 单次裁决」；预研「投票 / 多轮仲裁」时不破坏现有 `run_pipeline` 契约，可新增 `run_pipeline_v2` 或 profile。
4. **安全与成本**：租户级 token 预算、模型路由（见 highlevel「可控性」）；与 **API_CONTRACT** 平台 metrics 对齐。
5. **与 Step0 关系**：日志声明 CSV 为 Step0 SSOT；YAML 关键词合并策略见 [03_metadata_extraction.md](03_metadata_extraction.md)。

---

## 4. 目标形态与接口（本模块对外承诺）

- **输入**：`ocr_text`、Step0 hits（dict 或 str 列表）、可选 `config` / `rules_context`。
- **输出**：与 **04_ai_pipeline_design** 及现有 `AIResult` 映射字段对齐的 dict（含 `reasoning_trace` 等内部追溯）。
- **扩展**：新 Agent = 规则知识 + prompt 模板 + 模型配置，不修改核心裁决函数签名（优先配置驱动）。

---

## 5. 与其他模块的并行约定

| 关系 | 说明 |
|------|------|
| **我依赖** | [02](02_text_alignment_provenance.md)（evidence 几何/偏移）；[03](03_metadata_extraction.md)、[04](04_catalog_rule_engine.md)（输入字段与规则优先级） |
| **我提供** | 审核结论、子 Agent 原始输出、裁决解释 |
| **依赖我** | 人工复核 UI（**03_frontend_module_design**）、审计（**05_human_override**） |

**Mock**：对 `invoke_llm_json` 返回固定 JSON；`build_rules_pipeline_context` 使用测试 CSV。

---

## 6. 代码阅读建议

| 时长 | 路径 |
|------|------|
| ~5 分钟 | **04_ai_pipeline_design** 第 4 节（Step0–3 图） |
| ~30 分钟 | `pipeline.run_pipeline` + `_format_step0_hits_for_prompt`、`_run_single_agent` |
| ~半天 | `rules_loader` CSV 列含义；触发 AI 分析的 API service（`api/v1` + document 状态） |

---

## 7. 思考题与自测

1. Step2 中「无规则则不调用 LLM」的设计意图？对覆盖率有何影响？
2. 若 Step3 与 Step2 某 Agent 结论冲突，当前代码如何强制保守策略？是否应记录到审计？
3. 如何将「子 Agent 输出」完整存入 `AIResult` JSON 而不泄露过大 prompt？（裁剪/哈希策略）
