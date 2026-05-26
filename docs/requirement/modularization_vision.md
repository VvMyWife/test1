# 架构演进愿景：平台化模块拆分

> **状态**：📋 规划中（详见 [backend_modular_refactor_6b64915a.plan.md](../../.cursor/plans/backend_modular_refactor_6b64915a.plan.md)）
>
> **目标**：将当前单体 FastAPI 应用拆分为可独立开发、部署的微服务架构

---

## 为什么要平台化？

### 当前状态（单体应用）

```
backend/
├── app/
│   ├── api/v1/       # 所有路由
│   ├── services/     # 所有业务逻辑
│   ├── models/       # 所有 ORM 模型
│   ├── operators/    # 可复用算子 ⚠️
│   ├── pipeline/     # 编排运行时 ⚠️
│   ├── inference/    # AI 推理 ⚠️
│   └── tasks/        # Celery 任务
```

**问题**：
- 所有代码耦合在单一 FastAPI 应用中
- `operators/` 和 `pipeline/` 难以被其他项目复用
- 各组员修改同一代码库，冲突频繁
- 测试和部署互相影响

### 目标状态（模块化架构）

```
backend/
├── foundation/                    # ⭐ 可独立发布的 Python 包
│   ├── pyproject.toml            # 单一包 + optional extras
│   ├── platform_foundation/
│   │   ├── operators/            # 纯 Python，无 FastAPI/SQLAlchemy
│   │   ├── pipeline/             # 纯 Python 编排
│   │   ├── inference/            # 推理适配器
│   │   ├── rules_engine/         # 规则引擎
│   │   └── contracts/           # 共享 DTO
│   └── extras/                   # optional-dependencies
│       ├── vision               # paddleocr, torch, transformers
│       ├── document             # mineru, pymupdf
│       └── ray                 # ray
│
├── services/                     # 可独立部署的服务
│   ├── doc-service/             # 文档管理服务
│   ├── extraction-service/       # 抽取/OCR 服务
│   ├── rag-service/             # RAG 服务
│   ├── agent-service/           # Agent 服务
│   └── platform-api/            # API 网关（聚合尚未拆走的路由）
│
└── jobs/                         # 离线批处理（Celery 任务）
```

---

## 核心设计原则

### 1. Foundation 保持"纯 Python"

```python
# ✅ Foundation 可以依赖
import torch
import pydantic
from typing import Protocol

# ❌ Foundation 不能依赖
from fastapi import FastAPI       # Web 框架
from sqlalchemy import Column      # ORM
from celery import Celery         # 任务队列（部署 concern）
```

**原因**：Foundation 是纯能力库，可以在任何 Python 项目中使用，不绑定 Web 框架或数据库。

### 2. 单向依赖

```
services/* → foundation (单向)
foundation → services (禁止)
```

### 3. ExecutionContextProvider Protocol

`ExecutionContext` 属于 Service 层，Foundation 通过 Protocol 接口获取执行信息：

```python
# foundation/pipeline/contracts.py
class ExecutionContextProvider(Protocol):
    def get_run_id(self) -> str: ...
    def get_tenant_id(self) -> str: ...
    def get_config_version(self) -> str: ...

# foundation/pipeline/runtime.py
def run_pipeline(ctx: OperatorContext, providers: ExecutionContextProvider):
    ...
```

### 4. Extras 分组

| Extra | 包含 | 使用场景 |
|-------|------|----------|
| `vision` | paddleocr, torch, transformers, doclayout-yolo | OCR、图像处理 |
| `document` | mineru, pymupdf, openpyxl | PDF 解析、文档处理 |
| `ray` | ray | 分布式执行 |
| *(queue)* | celery, redis | **不在 foundation**，放 services 层 |

---

## 服务拆分策略

### Phase 0：契约冻结（必须先做）

1. **ADR-001**：服务清单 + API Path 前缀约定
2. **ADR-002**：同步调用（HTTP）vs 异步（Celery/Redis）边界
3. **ADR-003**：数据库 schema/表前缀分域
4. **ADR-004**：服务发现策略（dev: docker-compose DNS；prod: Ingress）

### Phase A：抽 Foundation

1. 迁入无 DB/无 FastAPI 代码：`operators/`、`pipeline/contracts`、`pipeline/runtime`
2. 定义 `ExecutionContextProvider` Protocol
3. 整理 optional-dependencies

### Phase B：服务落地

按域从 `app/` 迁入 `services/<domain>/`：
- `doc-service`：文档上传、状态机
- `extraction-service`：OCR、布局解析
- `rag-service`：检索、RAG
- `agent-service`：Multi-Agent 审核
- `platform-api`：API 网关

---

## 对现有模块的影响

### `app/operators/` → `foundation/operators/`

| 现状 | 目标 |
|------|------|
| 在 FastAPI 应用内 | 迁入 `foundation/` 包 |
| 直接 import `app.inference` | 通过 `bridge_*` 解耦 |
| 算子内可能读取全局配置 | 通过 `OperatorContext.tags` 注入 |

### `app/pipeline/` → `foundation/pipeline/`

| 现状 | 目标 |
|------|------|
| `SqlAlchemyLineageRecorder` 在 pipeline 内 | PostgreSQL 实现在 service 层，Protocol 注入 |
| `ExecutionContext` 直接传入 | 通过 `ExecutionContextProvider` 接口获取 |

### `app/inference/` → `foundation/operators/*` 或 `foundation/infra/`

| 现状 | 目标 |
|------|------|
| `providers/` 在 `app/inference/` | 迁入 `foundation/infra/` |
| LLM 调用在 `app/inference/` | 作为 `foundation` extra 或 service |

---

## 团队分工建议（模块化后）

| 团队 | 职责 |
|------|------|
| **Foundation 组** | `foundation/` 包维护、extras、contracts Protocol |
| **文档/多模态组** | extraction-service（含 MinerU）、doc-service |
| **AI/Agent 组** | agent-service（含 Multi-Agent） |
| **检索/RAG 组** | rag-service |
| **平台工程组** | services 壳、DB 迁移、CICD、deploy |
| **离线组** | `jobs/`（Celery 任务） |

---

## 思考题

1. **Foundation 的边界**：如果一个算子需要访问数据库（如查询配置），应该放在 Foundation 还是 Service？
2. **版本同步**：当 Foundation 发布新版本时，各 Service 如何平滑升级？
3. **共享数据库**：多服务共享同一 PostgreSQL 实例时，schema 命名策略是什么？如何避免循环依赖？

---

## 参考资料

- [重构计划全文](../../.cursor/plans/backend_modular_refactor_6b64915a.plan.md)
- [Operators 文档](../OPERATORS.md)
- [Pipeline 文档](../PIPELINE.md)
- [跨模块契约](design/00_cross_module_contracts.md)
