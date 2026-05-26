# 设计：OCR 与结构化解析（MinerU 方向）

**对应需求**：[highlevel_direction.md](../highlevel_direction.md) 第 1 条。  
**主要负责人**：邵海洋（见 [ownership.md](../ownership.md)）。  
**前置阅读**：[00_cross_module_contracts.md](00_cross_module_contracts.md)；[docs/design/06_document_intelligence_platform.md](../../design/06_document_intelligence_platform.md)（任务阶段与存储）。

---

## 1. 文档目的与读者

- **初级**：先读 `00` 中坐标系与 `ArtifactRef`；再读本节「代码地图」按文件打开代码。
- **中级**：直接看「平台差距」与「目标形态」，做 MinerU 并发/GPU 与版式评估实验。

---

## 2. 现状：已实现能力与代码地图


| 能力                   | 说明                                                       | 代码 / 文档路径                                                                                                                                                                                                                     |
| -------------------- | -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| MinerU Python API 调用 | `do_parse`、产出 `middle.json`、超时与设备环境变量                    | [backend/app/services/mineru_parser.py](../../../backend/app/services/mineru_parser.py)                                                                                                                                       |
| 按页文本抽取               | 从 `middle.json` 聚合页面文本                                   | `extract_per_page_text_from_middle`、`iter_spans_from_middle`（同上文件）                                                                                                                                                            |
| 算子级封装                | `LayoutExtractResult`、`coord_space`                      | [backend/app/operators/layout_extract_mineru.py](../../../backend/app/operators/layout_extract_mineru.py)、[backend/app/operators/support/document_contracts.py](../../../backend/app/operators/support/document_contracts.py) |
| Celery 任务            | `mineru_layout_only`、`mineru_parse` 等阶段写入 layout、JobPage | [backend/app/tasks/celery_tasks.py](../../../backend/app/tasks/celery_tasks.py)                                                                                                                                               |
| 配置项                  | backend、parse_method、lang、timeout、device                 | [backend/app/core/config.py](../../../backend/app/core/config.py)（`Settings` 中 `mineru_`*）                                                                                                                                    |


**权威设计引用**：平台任务与内部 Job 阶段关系见 **[docs/design/06_document_intelligence_platform.md](../../design/06_document_intelligence_platform.md)**；算子与 pipeline 边界见 **[CLAUDE.md](../../../CLAUDE.md)**（`pipeline` → `operators` 单向）。

---

## 3. 与平台可共用能力的差距（优化方向）

1. **Batch / GPU**：当前以进程内调用为主；海量文档需要明确「每 worker 并发度、GPU 独占/共享、队列隔离」与 **Ray/Celery** 分工（详见 [05_platform_pipeline_ray_operators.md](05_platform_pipeline_ray_operators.md)）。
2. **结构化信息 SSOT**：`middle.json` 已含段落/行/span/bbox；需与 [02_text_alignment_provenance.md](02_text_alignment_provenance.md) 约定**同一套**页内偏移与 ID 策略，避免各模块自行拼接文本。
3. **版式能力评估**：表格、多栏、复杂排版需**样本集 + 通过标准**（可定位性、阅读顺序正确率），预研输出建议独立成表：场景 | MinerU 表现 | 是否阻塞上线。
4. **可观测性**：建议在阶段日志中统一 `layout_provider`、`page_count`、`middle_sha256`（或路径 + mtime），与 `platform_job` 的 `metrics` 字段对齐（**API_CONTRACT** §12.4）。
5. **失败语义**：`run_mineru` 返回 `False` 与异常路径需在平台层映射为**可重试** vs **永久失败**（与 `config_snapshot` 冻结版本相关）。

---

## 4. 目标形态与接口（本模块对外承诺）

- **输入**：本地可读文件路径 + MIME（可选）；租户/任务上下文来自 `OperatorContext`（见 [00](00_cross_module_contracts.md)）。
- **输出**：`LayoutExtractResult`：`pages_text[]`、`middle_json_path`、`page_count`、`coord_space`、`meta`（错误时 `meta.error`）。
- **不承诺**：在本模块内写业务库表；持久化由 Service / Celery 任务完成。

下游 **仅依赖** `middle.json` 路径与 `coord_space`，不依赖 MinerU 内部临时目录结构。

---

## 5. 与其他模块的并行约定


| 关系      | 说明                                                                                                              |
| ------- | --------------------------------------------------------------------------------------------------------------- |
| **我提供** | 每页文本、`middle.json`、坐标系标签                                                                                        |
| **我依赖** | 存储路径、任务调度（Celery/Ray）、`settings`                                                                                |
| **依赖我** | [02](02_text_alignment_provenance.md)、[07](07_sensitive_detection.md)（几何）、[08](08_retrieval_rag.md)（chunk 与页结构） |


**Mock**：单元测试可用小型 fixture `middle.json` + 固定 `pages_text`，无需安装 MinerU。

---

## 6. 代码阅读建议


| 时长     | 路径                                                                                          |
| ------ | ------------------------------------------------------------------------------------------- |
| ~5 分钟  | `document_contracts.LayoutExtractResult`、`layout_extract_mineru.extract_layout_with_mineru` |
| ~30 分钟 | `mineru_parser.run_mineru`、`iter_spans_from_middle`、`extract_per_page_text_from_middle`     |
| ~半天    | `celery_tasks.mineru_parse` / `mineru_layout_only` 与 `job_service` 写库流程；对照 **06 设计文档** 平台阶段 |


---

## 7. 思考题与自测

1. `f_dump_middle_json=True` 关闭时，下游哪些模块会失效？
2. MinerU 超时后任务应 `retry` 还是 `failed`？依据是什么（依赖可恢复 vs 输入损坏）？
3. 多租户下同一物理文件是否应共享一份 `middle.json` 缓存？与 `content-hash` 缓存策略如何结合（见平台 OCR cache 相关配置 **.env.example**）？

