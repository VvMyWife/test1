# 设计：分布式处理 Pipeline（Ray）与算子（Operator）标准化

**对应需求**：[highlevel_direction.md](../highlevel_direction.md) 第 5、6 条。  
**主要负责人**：总负责牵头（见 [ownership.md](../ownership.md)；本条在 ownership 表中未单列，与 **邵海洋 / 各模块 Owner** 联合评审）。  
**前置阅读**：[00_cross_module_contracts.md](00_cross_module_contracts.md)；**[docs/design/06_document_intelligence_platform.md](../../design/06_document_intelligence_platform.md)**。

---

## 1. 文档目的与读者

- **初级**：弄清 **Celery**（控制面）与 **Ray**（数据面）、**算子**（单步逻辑）三者的分工；会读 `build_image_pipeline_v1`。
- **中级**：设计阶段拆分、幂等、metrics、资源隔离，并与 `platform_job` 状态机对齐。

---

## 2. 现状：已实现能力与代码地图

| 能力 | 说明 | 代码 / 文档路径 |
|------|------|-----------------|
| 算子基类与 I/O | `BaseOperator`、`OperatorInput`/`OperatorOutput`、`ArtifactRef`、`OperatorError` | [docs/OPERATORS.md](../../../docs/OPERATORS.md) |
| 算子说明 | 组合方式与目录约定 | [docs/OPERATORS.md](../../../docs/OPERATORS.md)（完整文档） |
| 默认图片 DAG | OCR → 清洗 → 敏感 → 分级 → 结构化输出 | [backend/app/pipeline/factories.py](../../../backend/app/pipeline/factories.py) `build_image_pipeline_v1` |
| 同步逐步执行 | `run_image_pipeline_steps`、lineage 钩子 | [backend/app/pipeline/image_batch.py](../../../backend/app/pipeline/image_batch.py) |
| 线程池后端 | `LocalThreadPoolBackend` | [backend/app/pipeline/executors.py](../../../backend/app/pipeline/executors.py)（由 `runtime` 引用） |
| Ray 后端 | `RayBackend.map`、`try_create_ray_backend`、`enable_ray_pipeline` | [backend/app/pipeline/ray_runtime.py](../../../backend/app/pipeline/ray_runtime.py) |
| Ray 批路径 | `run_image_pipeline_batch_ray`、`ray_image_pipeline_item` | [backend/app/pipeline/image_batch.py](../../../backend/app/pipeline/image_batch.py) |
| 血缘记录 | `SqlAlchemyLineageRecorder`、`OperatorRunStep` 等 | [backend/app/pipeline/lineage_repository.py](../../../backend/app/pipeline/lineage_repository.py)、[backend/app/models/operator_lineage.py](../../../backend/app/models/operator_lineage.py) |
| 执行上下文 | 平台/Celery 边界 | [backend/app/execution/context.py](../../../backend/app/execution/context.py) |
| 平台任务 + 队列 | driver、`platform_ray_driver`、回退 `pipeline` | **06 设计文档**；[backend/app/celery_app.py](../../../backend/app/celery_app.py)（task routes） |

**导入约束**：业务代码应 `from app.pipeline import ...`，勿从 `app.operators` 再导出 pipeline（避免循环依赖，见 **CLAUDE.md**）。

---

## 3. Ray vs Celery（决策表 + 当前默认）

| 维度 | Celery | Ray（当前实现） |
|------|--------|-----------------|
| 主要职责 | 任务入队、重试、平台 driver、阶段编排 | 批内 `map` 并行（如图片 pipeline item） |
| 进程模型 | worker 常驻 | `ray.init` 连接集群或本地 |
| 可选性 | 默认开启 | `uv sync --extra ray` + `ENABLE_RAY_PIPELINE` 等配置 |
| 回退 | — | `try_create_ray_backend()` 失败则走线程池/ Celery 路径（见 `image_batch` 调用方） |

**预研结论**：不在此重复长篇对比；各阶段 owner 在评审会上补充「MinerU/LLM 重任务」是否单独 Ray Actor / 队列。文档级结论应回写 **06 设计文档** 或运维 runbook（**[docs/implementation/platform_capacity_runbook.md](../../implementation/platform_capacity_runbook.md)** 若存在）。

---

## 4. 与平台可共用能力的差距（优化方向）

1. **阶段粒度**：OCR → 解析 → 抽取 → 编目 → 质检 与现有 **JobStage** / **platform_job** 聚合视图对齐；每阶段输入输出引用 [00 `ArtifactRef`](00_cross_module_contracts.md)。
2. **幂等与重试**：统一 `OperatorError.retryable` 与 Celery `autoretry` 策略；平台侧 `payload_hash`、**API_CONTRACT** §12。
3. **可观测性**：结构化日志字段（`platform_job_id`、`trace_id`、`stage`、`latency_ms`）与 **06 设计文档** §4.2 一致；算子级 `execution_meta` 与 DB lineage 二选一或双写策略需明确。
4. **资源隔离**：GPU 任务与 CPU 扫描分队列；避免 Ray 与 MinerU 子进程在 worker 内 OOM 互杀。
5. **DAG-lite 落地**：`depends_on` 校验与真实拓扑一致（**06 设计文档** §5），防止配置宣称 DAG、实现仍是线性。

---

## 5. 目标形态与接口（本模块对外承诺）

- **算子**：无跨请求可变全局状态；`OperatorContext` 可序列化以便 Ray worker 重建（见 `contracts.py` 注释）。
- **编排**：仅 `pipeline` 包依赖 `operators`；算子不 import pipeline。
- **平台**：Worker **只读** `config_snapshot`；不在执行路径解析可变 ruleset。

---

## 6. 与其他模块的并行约定

| 关系 | 说明 |
|------|------|
| **我提供** | 执行后端抽象、默认 DAG 组装方式、血缘扩展点 |
| **依赖我** | [01](01_ocr_layout_mineru.md)（任务阶段挂载）、[03](03_metadata_extraction.md)–[07](07_sensitive_detection.md)（以算子或服务步骤接入） |

**Mock**：`MemoryLineageRecorder`、`run_image_pipeline_steps` + 假 `ImagePipelineState`。

---

## 7. 代码阅读建议

| 时长 | 路径 |
|------|------|
| ~5 分钟 | `factories.build_image_pipeline_v1`、`contracts.OperatorContext` |
| ~30 分钟 | `image_batch.run_image_pipeline_steps`、`base.BaseOperator.run_with_hooks` |
| ~半天 | `run_image_pipeline_batch_ray`、`platform_tasks` driver、`ExecutionContext` 注入路径 |

---

## 8. 思考题与自测

1. `ray_image_pipeline_item` 必须是顶层函数的原因？对算子热插拔有何限制？
2. 血缘写入失败时，是否应 fail whole job？与「可审计」目标如何权衡？
3. `enable_ray_pipeline=false` 时，平台 SLA 与吞吐如何验证（见 loadtest **CLAUDE.md**）？
