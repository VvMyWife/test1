from __future__ import annotations

from collections.abc import Iterable as IterableABC
from collections.abc import Mapping
from dataclasses import dataclass
import time
from typing import Iterator, Literal, Optional, Self

# [核心契约层] 定义了数据流的协议，确保不同 Operator 之间能“听懂”对方的话
from .contracts import (
    ArrowStream,      # ⚡️ 高性能列式流：Iterator[pa.RecordBatch]
    HAS_ARROW,        # 🛠 环境检查：判断当前系统是否安装了 pyarrow
    Item,             # 📦 基础数据单元：通常是 Dict 或 Pydantic 对象
    ItemStream,       # 🌊 逐条对象流：Iterator[Item]
    OperatorContext,  # 📋 上下文：携带 TraceID、配置、资源句柄等
    OperatorError,    # ⚠️ 框架标准异常：支持重试标记和错误码
    OperatorSchema,   # 📐 契约声明：描述输入输出的数据结构和格式
    RetryPolicy,      # 🔄 重试策略：定义退避算法、最大次数等
    TransformPlan,    # 🔄 转换计划：当上下游格式不匹配时的“胶水”逻辑
)

if HAS_ARROW:  # pragma: no cover
    import pyarrow as pa  # type: ignore

# [并行模型] sequential:单线程 | thread:多线程(IO密集型) | process:多进程(计算密集型)
ConcurrencyModel = Literal["sequential", "thread", "process", "distributed"]


class OperatorHooks:
    """
    [监控钩子] 开发者通过实现此接口来接入日志、链路追踪(Tracing)和指标监控(Metrics)
    """
    def on_start(self, ctx: OperatorContext, *, op_name: str, op_version: str) -> None: ...

    def on_end(
        self,
        ctx: OperatorContext,
        *,
        op_name: str,
        op_version: str,
        status: Literal["ok", "error"],
        duration_ms: float,
    ) -> None: ...

    def on_error(
        self,
        ctx: OperatorContext,
        *,
        op_name: str,
        op_version: str,
        error: BaseException,
    ) -> None: ...


@dataclass(frozen=True)
class BaseOperator:
    """
    [算子基类] 所有业务逻辑算子必须继承此类。
    
    开发者实现步骤：
    1. 实现 input_schema/output_schema 声明数据格式
    2. 根据需求重写 process_item (单条处理) 或 process_arrow_batch (批量处理)
    """
    op_name: str     # 算子唯一标识，建议使用下划线命名，如 "text_normalizer"
    op_version: str  # 版本号，用于灰度发布或逻辑追溯

    # 💡 确定性标记：如果输入相同则输出必相同，请设为 True，方便框架做缓存(Cache)
    is_deterministic: bool = True
    # 💡 副作用标记：是否会修改外部数据库、发送邮件等。影响重试策略
    has_side_effects: bool = False
    concurrency_model: ConcurrencyModel = "thread"
    retry_policy: RetryPolicy = RetryPolicy()

    hooks: Optional[OperatorHooks] = None

    @classmethod
    def from_config(cls, config: dict) -> Self:
        return cls(**config)  # type: ignore[arg-type]

    def input_schema(self) -> Optional[OperatorSchema]:
        """[需重写] 声明该算子期待接收的数据格式。返回 None 表示接受任何格式。"""
        return None

    def output_schema(self) -> Optional[OperatorSchema]:
        """[需重写] 声明该算子输出的数据格式。帮助框架在运行前发现 Pipeline 错误。"""
        return None

    def can_accept(self, upstream: OperatorSchema) -> bool:
        expected = self.input_schema()
        if expected is None:
            return True
        return expected.kind == upstream.kind and expected.schema == upstream.schema

    def plan_transform(self, upstream: OperatorSchema) -> Optional[TransformPlan]:
        if self.can_accept(upstream):
            return None
        return TransformPlan(kind="custom", description="No automatic transform available")

    def supports_item(self) -> bool:
        """检查子类是否重写了 process_item。框架以此判断是否能走逐条处理路径。"""
        return self.process_item.__func__ is not BaseOperator.process_item  # type: ignore[attr-defined]

    def supports_arrow(self) -> bool:
        """检查子类是否重写了 process_arrow_batch。框架以此判断是否支持高性能路径。"""
        return self.process_arrow_batch.__func__ is not BaseOperator.process_arrow_batch  # type: ignore[attr-defined]

    def process(
        self,
        ctx: OperatorContext,
        input_stream: ItemStream | ArrowStream,
        *,
        path: Literal["item", "arrow"],
    ) -> ItemStream | ArrowStream:
        """
        [主入口] 框架调度中心。负责：生命周期管理、路径分发、异常捕获和耗时统计。
        🚫 请勿在子类中重写此方法！
        """
        start = time.perf_counter()
        status: Literal["ok", "error"] = "ok"
        self._hook_start(ctx)
        try:
            # 路径 A: Arrow 批量路径 (通过 pyarrow 处理，适合大数据量)
            if path == "arrow":
                if not HAS_ARROW:
                    raise OperatorError(
                        "pyarrow is not installed; Arrow path is unavailable",
                        code="ARROW_NOT_INSTALLED",
                        retryable=False,
                    )
                if not self.supports_arrow():
                    raise OperatorError(
                        f"{self.op_name} does not support Arrow path",
                        code="UNSUPPORTED_PATH",
                        retryable=False,
                    )
                out = self._process_arrow_stream(ctx, input_stream)  # type: ignore[arg-type]
            # 路径 B: Item 逐条路径 (适合 LLM 交互、复杂 Python 逻辑)
            elif path == "item":
                if not self.supports_item():
                    raise OperatorError(
                        f"{self.op_name} does not support item path",
                        code="UNSUPPORTED_PATH",
                        retryable=False,
                    )
                out = self._process_item_stream(ctx, input_stream)  # type: ignore[arg-type]
            else:  # pragma: no cover
                raise AssertionError(f"unknown path: {path}")
            return out
        except BaseException as e:
            status = "error"
            self._hook_error(ctx, e)
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000.0
            self._hook_end(ctx, status=status, duration_ms=duration_ms)

    def process_item(self, ctx: OperatorContext, item: Item) -> Item | IterableABC[Item] | None:
        """
        [业务核心 A] 逐条处理逻辑。
        @param item: 传入的数据对象
        @return: 处理后的数据对象，或多个数据对象（用于 document -> page fan-out）
        """
        raise NotImplementedError

    def process_arrow_batch(self, ctx: OperatorContext, batch: "pa.RecordBatch") -> "pa.RecordBatch":
        """
        [业务核心 B] 批量处理逻辑。
        @param batch: Apache Arrow 的列式内存块，包含多行数据
        @return: 处理后的 RecordBatch
        """
        raise NotImplementedError

    def _process_item_stream(self, ctx: OperatorContext, input_stream: ItemStream) -> ItemStream:
        """[流化包装] 将逐条处理逻辑包装成迭代器，保持内存占用的平稳"""
        def gen() -> Iterator[Item]:
            for item in input_stream:
                result = self.process_item(ctx, item)
                yield from self._iter_item_results(result)

        return gen()

    def _process_arrow_stream(self, ctx: OperatorContext, input_stream: ArrowStream) -> ArrowStream:
        """[流化包装] 批量搬运 RecordBatch，实现高性能列式流处理"""
        def gen() -> Iterator["pa.RecordBatch"]:
            for batch in input_stream:
                yield self.process_arrow_batch(ctx, batch)

        return gen()

    def _iter_item_results(self, result: Item | IterableABC[Item] | None) -> Iterator[Item]:
        if result is None:
            return

        if isinstance(result, Mapping):
            yield result
            return

        if isinstance(result, IterableABC):
            for index, sub_item in enumerate(result):
                if not isinstance(sub_item, Mapping):
                    raise OperatorError(
                        f"{self.op_name} emitted non-mapping item at index {index}",
                        code="INVALID_ITEM_OUTPUT",
                        retryable=False,
                    )
                yield sub_item
            return

        raise OperatorError(
            f"{self.op_name} emitted unsupported item output type: {type(result).__name__}",
            code="INVALID_ITEM_OUTPUT",
            retryable=False,
        )

    def _hook_start(self, ctx: OperatorContext) -> None:
        if self.hooks is not None:
            self.hooks.on_start(ctx, op_name=self.op_name, op_version=self.op_version)

    def _hook_error(self, ctx: OperatorContext, error: BaseException) -> None:
        if self.hooks is not None:
            self.hooks.on_error(ctx, op_name=self.op_name, op_version=self.op_version, error=error)

    def _hook_end(self, ctx: OperatorContext, *, status: Literal["ok", "error"], duration_ms: float) -> None:
        if self.hooks is not None:
            self.hooks.on_end(
                ctx,
                op_name=self.op_name,
                op_version=self.op_version,
                status=status,
                duration_ms=duration_ms,
            )
