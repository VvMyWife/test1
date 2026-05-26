# 算子库 (Operators)

可复用的**能力单元**，与**编排运行时** `app.pipeline` 分离。算子只管输入/输出与 `BaseOperator` 契约；批跑、分块、并发、Ray、血缘在 `app.pipeline`。

---

## 核心概念

### 算子 (Operator)
无状态单步处理单元，如 OCR、文本清洗、敏感检测、风险分类等。每个算子实现 `BaseOperator` 接口。

### 算子链 (Pipeline)
多个算子按顺序组合，由 `app.pipeline` 负责执行和调度。

### 算子上下文 (OperatorContext)
传递执行信息（trace_id, run_id, batch_id, tenant_id, config_version），算子通过上下文获取执行信息。

---

## 目录结构

```
backend/app/operators/
├── base.py                    # BaseOperator 抽象类
├── contracts.py                # OperatorContext, OperatorError
├── image_pipeline_models.py    # 图片批处理链共用 Pydantic 状态
├── extract_text_from_image.py  # OCR 算子
├── clean_ocr_text.py          # 文本清洗算子
├── layout_extract_mineru.py   # MinerU PDF 布局解析算子
├── scan_rules_hits.py         # 规则命中算子
├── sensitive_geometry_from_layout.py  # 敏感信息几何定位算子
├── classify_archive_risk.py    # 风险分类算子
├── static_lexicon_merge.py    # 静态词库合并算子
├── structure_pipeline_output.py # 结构化输出算子
├── document_review_pipeline.py # 文档级审查流水线
├── support/                   # 桥接层
│   ├── bridge_ocr.py          # 调用 inference OCR
│   ├── bridge_rules.py       # 调用 rules_engine
│   └── document_contracts.py  # 文档 DTO
└── README.md                  # 本文档（代码内）
```

---

## 快速开始

### 构建并运行图片批处理流水线

```python
from app.operators.contracts import OperatorContext
from app.pipeline import (
    LocalThreadPoolBackend,
    PipelineBatchConfig,
    build_image_pipeline_v1,
    run_image_pipeline_batch,
)
from app.pipeline.schemas import ImageBatchItem

ctx = OperatorContext(
    trace_id="trace-1",
    run_id="run-1",
    batch_id="batch-1",
    item_id="placeholder",
    tenant_id="tenant-1",
    config_version="v1",
)
items = [ImageBatchItem(item_id="img-1", image_path="/path/to/a.png")]
ops = build_image_pipeline_v1()
results = run_image_pipeline_batch(
    ctx,
    items,
    ops,
    backend=LocalThreadPoolBackend(default_workers=None),
    cfg=PipelineBatchConfig(chunk_size=64, max_in_flight=32),
    lineage=None,
)
```

### 文档级算子

与单图批处理链并列，面向整份 PDF/图片：

```python
from app.operators.layout_extract_mineru import extract_layout_with_mineru
from app.operators.static_lexicon_merge import merge_static_lexicon_hits
from app.operators.sensitive_geometry_from_layout import localize_static_hits_in_middle
from app.operators.document_review_pipeline import run_static_geometry_pipeline
```

---

## 并发与批策略

| 参数 | 说明 |
|------|------|
| `PipelineBatchConfig.chunk_size` | 外层分批提交窗口（限制单次 map 任务数） |
| `PipelineBatchConfig.max_in_flight` | 期望并发上限 |
| `LocalThreadPoolBackend(default_workers=None)` | 默认不额外限制线程数 |

### 重试机制
- 单步重试：`BaseOperator.max_retries` + `OperatorError(retryable=True)`
- 仅对 `OperatorError` 且 `retryable=True` 指数退避重试
- 普通异常与 `retryable=False` 不重试

---

## Ray 分布式执行（可选）

```bash
# 1. 安装
uv sync --extra ray

# 2. 环境配置
ENABLE_RAY_PIPELINE=true
RAY_ADDRESS=ray://...  # 可选
```

- Ray worker 通过流水线名称 **`image_v1`** 在进程内重建算子链，避免 pickle 算子实例
- 显式设置 `RAY_ADDRESS` 时，连接失败会直接抛错

---

## 血缘落库

使用 `SqlAlchemyLineageRecorder(session)` 作为 lineage 参数：

```python
from app.pipeline.lineage_repository import SqlAlchemyLineageRecorder

recorder = SqlAlchemyLineageRecorder(session)
results = run_image_pipeline_batch(ctx, items, ops, lineage=recorder)
```

在业务 `Session` 提交事务后写入 `operator_runs` / `operator_run_steps`。

---

## 依赖关系

```
pipeline → operators → inference / rules_engine
```

- **OCR**: `support/bridge_ocr.py` → `app.inference.get_vision_ocr_provider`
- **敏感规则扫描**: `support/bridge_rules.py` → `app.rules_engine.scan_with_rules`

---

## 日志与排障

### 日志级别
- 全局：`LOG_LEVEL`（默认 INFO）
- 算子+编排：`OPERATORS_LOG_LEVEL`（可选覆盖）

### 日志内容约定
| 级别 | 内容 |
|------|------|
| INFO | `batch_start` / `batch_done`（含 run_id, batch_id）；`item_fail` |
| DEBUG | `step_ok`（含 step_key, 耗时, retry_count）；分块信息 |

### 检索
```bash
grep "run_id=" logs/*.log
grep "step_ok" logs/*.log
```

---

## 配置注入

算子减少对 `app/core/config` 的全局读取，改为显式 config / `OperatorContext.tags` 传递。

---

## 后续接入

当前**未**接入 `ai_service` / Celery。接入时建议：

- Celery/Ray driver 仅负责批调度与 `ExecutionContext` 传递
- 算子库保持无状态，租户/规则版本写入 `OperatorContext` 与 `operator_runs.config_version`
