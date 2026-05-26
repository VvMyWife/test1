# 设计：Data Agent（NL2Query 与数据分析）

**对应需求**：[highlevel_direction.md](../highlevel_direction.md) 第 10 条。  
**主要负责人**：徐仕林（见 [ownership.md](../ownership.md)）。  
**前置阅读**：[00_cross_module_contracts.md](00_cross_module_contracts.md)；**[docs/API_CONTRACT.md](../../API_CONTRACT.md)**（文档列表、平台任务、审计）；[08_retrieval_rag.md](08_retrieval_rag.md)。

---

## 1. 文档目的与读者

- **初级**：理解本能力在仓库中**尚无专门模块**；先掌握现有只读 API 能回答哪些「统计类」问题。
- **中级**：设计语义层（指标目录）、NL2SQL/规划、Tool-calling 安全边界。

---

## 2. 现状：已实现能力与代码地图

| 能力 | 说明 | 代码 / 文档路径 |
|------|------|-----------------|
| 文档维度查询 | 列表过滤、分页、总数 | **API_CONTRACT** §4.2 |
| 文档详情 | 单条元数据与关联 | **API_CONTRACT** §4.3 |
| AI 分析结果 | 触发与读取 | **API_CONTRACT** §5 |
| 批量/平台任务 | 异步任务状态 | **API_CONTRACT** §7、§12 |
| 审计日志 | 按文档查询 | **API_CONTRACT** §8.1 |
| 档案层级 | 户/卷/件/页 | **API_CONTRACT** §3.4 |

**自然语言 → 查询计划 → 执行 → 解释**：**当前代码库无** Data Agent、NL2SQL、或统一「指标语义层」服务；属 **MVP 之后** 能力。本设计文档用于**预研与接口预留**，避免与现有安全模型冲突。

---

## 3. 与平台可共用能力的差距（优化方向）

1. **语义层（关键）**：将「归档数量」「某类档案」「异常档案（缺字段/低质量）」定义为 **MetricDefinition**（口径、SQL/ES 片段、权限范围），禁止 Agent 随意拼 SQL。
2. **Tool 边界**：只暴露 **白名单 Tools**（如 `run_metric_query(id, filters)`、`search_documents(structured_filters)`），而非任意 `execute_sql`；与 **租户隔离** 强绑定。
3. **NL2Query 路径**：Text2SQL 仅针对 **受控视图**（非裸表）；复杂问题用 Planner-Executor 多步，但每步工具仍受限。
4. **与 RAG 结合**：分析类问题可能需检索案例（见 [08](08_retrieval_rag.md)）；回答须标注数据来源（API 结果 vs 检索片段）。
5. **可审计**：记录自然语言问题、解析后的计划、执行参数与结果摘要（脱敏），写入审计或专用 `data_agent_runs` 表（待建模）。

---

## 4. 目标形态与接口（本模块对外承诺）

- **首期**：设计文档 + POC；**不修改**现有 **API_CONTRACT** 除非评审通过新增 `POST /api/v1/data-agent/query` 类端点。
- **输入**：自然语言 + `tenant_id` + 可选 `database_id` 范围。
- **输出**：结构化结果（表格/聚合）+ 自然语言解释 + **引用**（用了哪些指标定义、哪些过滤条件）。

---

## 5. 与其他模块的并行约定

| 关系 | 说明 |
|------|------|
| **我依赖** | [08](08_retrieval_rag.md)（若 Tool 含检索）；**DATABASE_SCHEMA**（只读视图设计） |
| **我提供** | 管理端/运营端分析能力；不替代业务审阅三栏核心流程 |
| **依赖我** | 无强依赖；各模块可继续独立交付 |

**Mock**：固定 JSON「指标目录」+ 假查询结果，测 Planner 不连真实 DB。

---

## 6. 代码阅读建议

| 时长 | 路径 |
|------|------|
| ~5 分钟 | **API_CONTRACT** §4.2、§9.1 状态枚举 |
| ~30 分钟 | 后端 `documents`/`databases` router 与 service（了解真实过滤字段） |
| ~半天 | **[docs/DATABASE_SCHEMA.md](../../DATABASE_SCHEMA.md)** + 草拟只读 SQL 视图清单 |

---

## 7. 思考题与自测

1. 若 Agent 生成 `DELETE` 或 `UPDATE`，如何在架构层**不可能**发生？
2. 「今年某类档案数量」中的「今年」以 `created_at` 还是「归档日期」字段为准？语义层如何解决歧义？
3. 低质量档案的定义是否应配置化（规则引擎 [04](04_catalog_rule_engine.md)）以便 Data Agent 只消费统一口径？
