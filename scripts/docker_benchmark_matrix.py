#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import time
import zipfile
from html import escape
from typing import Any


DEFAULT_ENGINES = ("ocr", "paddle")
DEFAULT_CONCURRENCY_VALUES = (1, 2, 4, 8, 12, 16, 24, 32)
DEFAULT_API_CONCURRENCY_VALUES = (1, 2, 4, 8, 12, 16, 24, 32, 64, 128)
STATE_VERSION = 2


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    host_output_root = (repo_root / args.host_output_root).resolve()
    host_output_root.mkdir(parents=True, exist_ok=True)

    state_path = Path(args.state_path).expanduser().resolve() if args.state_path else host_output_root / "benchmark_state.json"
    excel_path = Path(args.excel_path).expanduser().resolve() if args.excel_path else host_output_root / "docker_benchmark.xlsx"
    state = load_state(state_path, reset=args.reset_state)

    engines = parse_engines(args.engines)
    concurrency_values = parse_int_list(args.concurrency_values)
    shared_api_values = parse_int_list(args.api_concurrency_values)
    mineru_api_values = (
        parse_int_list(args.mineru_api_concurrency_values)
        if args.mineru_api_concurrency_values
        else shared_api_values
    )
    paddle_api_values = (
        parse_int_list(args.paddle_api_concurrency_values)
        if args.paddle_api_concurrency_values
        else shared_api_values
    )

    print(f"repo_root={repo_root}", flush=True)
    print(f"host_output_root={host_output_root}", flush=True)
    print(f"state_path={state_path}", flush=True)
    print(f"excel_path={excel_path}", flush=True)
    print(f"engines={engines}", flush=True)
    print(f"concurrency_values={concurrency_values}", flush=True)
    print(f"mineru_api_concurrency_values={mineru_api_values}", flush=True)
    print(f"paddle_api_concurrency_values={paddle_api_values}", flush=True)

    for engine in engines:
        api_pairs = build_api_pairs(
            engine=engine,
            mineru_api_values=mineru_api_values,
            paddle_api_values=paddle_api_values,
        )
        for mineru_api_max, paddle_api_max in api_pairs:
            pending_values = pending_concurrency_values(
                state,
                engine=engine,
                mineru_api_max=mineru_api_max,
                paddle_api_max=paddle_api_max,
                concurrency_values=concurrency_values,
                continue_after_failure=args.continue_after_failure,
                rerun_failed=args.rerun_failed,
            )
            if not pending_values:
                continue

            restart_services(
                repo_root=repo_root,
                engine=engine,
                mineru_api_max=mineru_api_max,
                paddle_api_max=paddle_api_max,
                build=args.build,
                startup_timeout_seconds=args.startup_timeout_seconds,
            )

            for concurrency in pending_values:
                if (
                    not args.continue_after_failure
                    and not args.rerun_failed
                    and has_boundary_failure(
                        state,
                        engine=engine,
                        mineru_api_max=mineru_api_max,
                        paddle_api_max=paddle_api_max,
                    )
                ):
                    break
                trial = run_trial(
                    repo_root=repo_root,
                    engine=engine,
                    mineru_api_max=mineru_api_max,
                    paddle_api_max=paddle_api_max,
                    concurrency=concurrency,
                    input_dir=args.input_dir,
                    host_output_root=host_output_root,
                    container_output_root=args.container_output_root,
                    timeout_seconds=args.batch_timeout_seconds,
                )
                record_trial(state, trial)
                save_state(state_path, state)
                write_excel(excel_path, state)
                print_trial(trial)
                if trial["status"] != "success" and not args.continue_after_failure:
                    break

    save_state(state_path, state)
    write_excel(excel_path, state)
    print(f"done: {excel_path}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    script_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the Dockerized MinerU operator across table_engine, caller "
            "concurrency, and API server concurrency. Results are resumable and "
            "written to an .xlsx file."
        )
    )
    parser.add_argument("--repo-root", default=str(script_root), help="Project root containing docker-compose.yml.")
    parser.add_argument(
        "--engines",
        default="ocr,paddle",
        help="Comma-separated table engines. Allowed values: ocr,paddle.",
    )
    parser.add_argument(
        "--concurrency-values",
        default=",".join(str(item) for item in DEFAULT_CONCURRENCY_VALUES),
        help="Comma-separated caller concurrency values to test.",
    )
    parser.add_argument(
        "--api-concurrency-values",
        default=",".join(str(item) for item in DEFAULT_API_CONCURRENCY_VALUES),
        help=(
            "Backward-compatible shorthand for both MinerU and Paddle API max "
            "concurrency values when the more specific options are not set."
        ),
    )
    parser.add_argument(
        "--mineru-api-concurrency-values",
        default=None,
        help="Comma-separated MINERU_API_MAX_CONCURRENT_REQUESTS values to test.",
    )
    parser.add_argument(
        "--paddle-api-concurrency-values",
        default=None,
        help=(
            "Comma-separated PADDLE_TABLE_API_MAX_CONCURRENT_EXTRACTS values to "
            "test in paddle mode. Ignored for ocr mode."
        ),
    )
    parser.add_argument("--input-dir", default="/workspace/input", help="Container path for input PDFs.")
    parser.add_argument(
        "--host-output-root",
        default="output/docker_benchmark",
        help="Host-side benchmark output root, relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--container-output-root",
        default="/workspace/output/docker_benchmark",
        help="Container-side benchmark output root matching --host-output-root bind mount.",
    )
    parser.add_argument("--state-path", default=None, help="Optional explicit JSON state path.")
    parser.add_argument("--excel-path", default=None, help="Optional explicit .xlsx result path.")
    parser.add_argument(
        "--batch-timeout-seconds",
        type=int,
        default=3600,
        help="Per-trial timeout. This only wraps the batch command, not Docker service startup.",
    )
    parser.add_argument(
        "--startup-timeout-seconds",
        type=int,
        default=1800,
        help="Service health wait timeout after Docker recreate. This time is not counted in benchmark metrics.",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Use docker compose up --build when recreating services. Default is --no-build.",
    )
    parser.add_argument(
        "--continue-after-failure",
        action="store_true",
        help="Do not stop a fixed engine/API-max series after the first failed concurrency value.",
    )
    parser.add_argument("--rerun-failed", action="store_true", help="Retry failed trials already present in state.")
    parser.add_argument("--reset-state", action="store_true", help="Ignore and overwrite previous state.")
    return parser.parse_args()


def parse_engines(raw: str) -> list[str]:
    engines = [item.strip().lower() for item in raw.split(",") if item.strip()]
    invalid = [item for item in engines if item not in DEFAULT_ENGINES]
    if invalid:
        raise SystemExit(f"invalid engines: {invalid}; allowed: {DEFAULT_ENGINES}")
    return engines


def parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for item in raw.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        parsed = int(stripped)
        if parsed <= 0:
            raise SystemExit(f"concurrency values must be positive: {raw}")
        values.append(parsed)
    return sorted(dict.fromkeys(values))


def build_api_pairs(
    *,
    engine: str,
    mineru_api_values: list[int],
    paddle_api_values: list[int],
) -> list[tuple[int, int | None]]:
    if engine == "ocr":
        return [(mineru_api_max, None) for mineru_api_max in mineru_api_values]
    return [
        (mineru_api_max, paddle_api_max)
        for mineru_api_max in mineru_api_values
        for paddle_api_max in paddle_api_values
    ]


def load_state(path: Path, *, reset: bool) -> dict[str, Any]:
    if reset or not path.exists():
        return {
            "version": STATE_VERSION,
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "trials": {},
        }
    with path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    if state.get("version") != STATE_VERSION:
        raise SystemExit(f"unsupported benchmark state version: {state.get('version')}")
    state.setdefault("trials", {})
    return state


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = now_iso()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def pending_concurrency_values(
    state: dict[str, Any],
    *,
    engine: str,
    mineru_api_max: int,
    paddle_api_max: int | None,
    concurrency_values: list[int],
    continue_after_failure: bool,
    rerun_failed: bool,
) -> list[int]:
    pending: list[int] = []
    for concurrency in concurrency_values:
        key = trial_key(engine, mineru_api_max, paddle_api_max, concurrency)
        existing = state["trials"].get(key)
        if existing is not None:
            if existing.get("status") == "success" or not rerun_failed:
                if existing.get("boundary_failure") and not continue_after_failure:
                    break
                continue
        if (
            not continue_after_failure
            and not rerun_failed
            and has_boundary_failure(
                state,
                engine=engine,
                mineru_api_max=mineru_api_max,
                paddle_api_max=paddle_api_max,
            )
        ):
            break
        pending.append(concurrency)
    return pending


def restart_services(
    *,
    repo_root: Path,
    engine: str,
    mineru_api_max: int,
    paddle_api_max: int | None,
    build: bool,
    startup_timeout_seconds: int,
) -> None:
    env = os.environ.copy()
    env["ENABLE_PADDLE_API"] = "true"
    env["MINERU_API_MAX_CONCURRENT_REQUESTS"] = str(mineru_api_max)
    env["PADDLE_TABLE_API_MAX_CONCURRENT_EXTRACTS"] = (
        str(paddle_api_max) if engine == "paddle" and paddle_api_max is not None else ""
    )
    env.setdefault("PADDLE_TABLE_API_STARTUP_TIMEOUT_SECONDS", str(startup_timeout_seconds))

    compose_cmd = ["docker", "compose", "up", "-d", "--force-recreate"]
    compose_cmd.append("--build" if build else "--no-build")
    compose_cmd.append("mineru-operator")
    print(
        f"recreate docker: engine={engine} mineru_api_max={mineru_api_max} "
        f"paddle_api_max={env['PADDLE_TABLE_API_MAX_CONCURRENT_EXTRACTS'] or 'unlimited'}",
        flush=True,
    )
    run_command(compose_cmd, cwd=repo_root, env=env, timeout=startup_timeout_seconds + 300)
    wait_services(repo_root=repo_root, engine=engine, timeout_seconds=startup_timeout_seconds)


def wait_services(*, repo_root: Path, engine: str, timeout_seconds: int) -> None:
    urls = ["http://127.0.0.1:8000/health"]
    if engine == "paddle":
        urls.append("http://127.0.0.1:8200/health")
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        ok = True
        for url in urls:
            cmd = [
                "docker",
                "compose",
                "exec",
                "-T",
                "mineru-operator",
                "curl",
                "-fsS",
                url,
            ]
            result = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True)
            if result.returncode != 0:
                ok = False
                last_error = (result.stderr or result.stdout or "").strip()
                break
        if ok:
            print(f"services ready for {engine}", flush=True)
            return
        time.sleep(2)
    raise SystemExit(f"services not ready after {timeout_seconds}s: {last_error}")


def run_trial(
    *,
    repo_root: Path,
    engine: str,
    mineru_api_max: int,
    paddle_api_max: int | None,
    concurrency: int,
    input_dir: str,
    host_output_root: Path,
    container_output_root: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    started_at = now_iso()
    if engine == "paddle":
        run_name = f"{engine}_mineru{mineru_api_max}_paddle{paddle_api_max}_c{concurrency}"
    else:
        run_name = f"{engine}_mineru{mineru_api_max}_c{concurrency}"
    container_output_dir = f"{container_output_root.rstrip('/')}/{run_name}"
    host_output_dir = host_output_root / run_name
    host_report_path = host_output_dir / "batch_report.json"

    env_items = [
        f"TABLE_ENGINE={engine}",
        f"CONCURRENCY={concurrency}",
        f"INPUT_DIR={input_dir}",
        f"OUTPUT_DIR={container_output_dir}",
        "MINERU_API_URL=http://127.0.0.1:8000",
    ]
    if engine == "paddle":
        env_items.append("PADDLE_TABLE_API_URL=http://127.0.0.1:8200")

    cmd = [
        "docker",
        "compose",
        "exec",
        "-T",
        "mineru-operator",
        "env",
        *env_items,
        "python",
        "/opt/mineru_workspace/scripts/extract_pdf_dir_minimal.py",
    ]
    print(f"trial start: {run_name}", flush=True)
    completed = subprocess.run(
        cmd,
        cwd=repo_root,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
    )
    stdout_tail = tail_text(completed.stdout)
    stderr_tail = tail_text(completed.stderr)
    report = load_report(host_report_path)
    status = determine_status(completed.returncode, report)
    error = ""
    if status != "success":
        error = extract_error(report) or stderr_tail or stdout_tail or f"return_code={completed.returncode}"

    return {
        "key": trial_key(engine, mineru_api_max, paddle_api_max, concurrency),
        "engine": engine,
        "api_max_concurrency": mineru_api_max,
        "mineru_api_max_concurrency": mineru_api_max,
        "paddle_api_max_concurrency": paddle_api_max if engine == "paddle" else None,
        "concurrency": concurrency,
        "status": status,
        "boundary_failure": status != "success",
        "started_at": started_at,
        "finished_at": now_iso(),
        "return_code": completed.returncode,
        "output_dir": str(host_output_dir),
        "container_output_dir": container_output_dir,
        "batch_report_path": str(host_report_path) if host_report_path.exists() else None,
        "total_elapsed_seconds": report.get("total_elapsed_seconds") if report else None,
        "page_count": report.get("page_count") if report else None,
        "pages_per_second": report.get("pages_per_second") if report else None,
        "pdf_count": report.get("pdf_count") if report else None,
        "success_count": report.get("success_count") if report else None,
        "failure_count": report.get("failure_count") if report else None,
        "skipped_count": report.get("skipped_count") if report else None,
        "table_engine": report.get("table_engine", engine) if report else engine,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "error": error,
    }


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        raise SystemExit(
            "command failed\n"
            f"cmd={' '.join(cmd)}\n"
            f"stdout={tail_text(result.stdout)}\n"
            f"stderr={tail_text(result.stderr)}"
        )
    return result


def load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        return {"_report_read_error": str(exc)}


def determine_status(return_code: int, report: dict[str, Any]) -> str:
    if return_code != 0:
        return "failed"
    if not report:
        return "failed"
    if report.get("_report_read_error"):
        return "failed"
    if int(report.get("failure_count") or 0) > 0:
        return "failed"
    if int(report.get("success_count") or 0) <= 0:
        return "failed"
    return "success"


def extract_error(report: dict[str, Any]) -> str:
    if not report:
        return ""
    if report.get("_report_read_error"):
        return str(report["_report_read_error"])
    failed_items = [
        item
        for item in report.get("items", [])
        if isinstance(item, dict) and not item.get("success", False)
    ]
    if not failed_items:
        return ""
    first = failed_items[0]
    return str(first.get("error") or first.get("error_type") or first)


def record_trial(state: dict[str, Any], trial: dict[str, Any]) -> None:
    state["trials"][trial["key"]] = trial


def trial_key(
    engine: str,
    mineru_api_max: int,
    paddle_api_max: int | None,
    concurrency: int,
) -> str:
    paddle_part = paddle_api_max if paddle_api_max is not None else "none"
    return f"{engine}|mineru_api={mineru_api_max}|paddle_api={paddle_part}|concurrency={concurrency}"


def has_boundary_failure(
    state: dict[str, Any],
    *,
    engine: str,
    mineru_api_max: int,
    paddle_api_max: int | None,
) -> bool:
    for trial in state.get("trials", {}).values():
        if (
            trial.get("engine") == engine
            and trial.get("mineru_api_max_concurrency") == mineru_api_max
            and trial.get("paddle_api_max_concurrency") == paddle_api_max
            and trial.get("boundary_failure")
            and trial.get("status") != "success"
        ):
            return True
    return False


def print_trial(trial: dict[str, Any]) -> None:
    print(
        "trial done: "
        f"{trial['key']} status={trial['status']} "
        f"mineru_api_max={trial.get('mineru_api_max_concurrency')} "
        f"paddle_api_max={trial.get('paddle_api_max_concurrency')} "
        f"elapsed={trial.get('total_elapsed_seconds')} "
        f"pages_per_second={trial.get('pages_per_second')} "
        f"pdf_count={trial.get('pdf_count')} "
        f"page_count={trial.get('page_count')}",
        flush=True,
    )
    if trial["status"] != "success":
        print(f"boundary failure recorded: {trial.get('error')}", flush=True)


def write_excel(path: Path, state: dict[str, Any]) -> None:
    trials = sorted_trials(state)
    best_by_engine = best_trials_by_engine(trials)
    sheets = [
        ("Summary", build_summary_rows(trials, best_by_engine)),
        ("OCR", build_engine_rows(trials, engine="ocr")),
        ("Paddle", build_engine_rows(trials, engine="paddle")),
    ]
    style_maps = {
        name: build_style_map(rows, best_by_engine, name)
        for name, rows in sheets
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml(len(sheets)))
        archive.writestr("_rels/.rels", root_rels_xml())
        archive.writestr("xl/workbook.xml", workbook_xml([name for name, _ in sheets]))
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml(len(sheets)))
        archive.writestr("xl/styles.xml", styles_xml())
        for index, (name, rows) in enumerate(sheets, start=1):
            archive.writestr(
                f"xl/worksheets/sheet{index}.xml",
                worksheet_xml(name, rows, style_maps[name]),
            )


def sorted_trials(state: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        state.get("trials", {}).values(),
        key=lambda item: (
            str(item.get("engine", "")),
            int(item.get("mineru_api_max_concurrency") or item.get("api_max_concurrency") or 0),
            int(item.get("paddle_api_max_concurrency") or 0),
            int(item.get("concurrency") or 0),
        ),
    )


def best_trials_by_engine(trials: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for trial in trials:
        if trial.get("status") != "success":
            continue
        elapsed = trial.get("total_elapsed_seconds")
        if elapsed is None:
            continue
        engine = str(trial.get("engine"))
        current = best.get(engine)
        if current is None:
            best[engine] = trial
            continue
        current_elapsed = current.get("total_elapsed_seconds")
        current_pps = current.get("pages_per_second") or 0
        trial_pps = trial.get("pages_per_second") or 0
        if elapsed < current_elapsed or (elapsed == current_elapsed and trial_pps > current_pps):
            best[engine] = trial
    return best


def build_summary_rows(
    trials: list[dict[str, Any]],
    best_by_engine: dict[str, dict[str, Any]],
) -> list[list[Any]]:
    rows = [
        [
            "engine",
            "best_status",
            "mineru_api_max_concurrency",
            "paddle_api_max_concurrency",
            "concurrency",
            "total_elapsed_seconds",
            "page_count",
            "pages_per_second",
            "pdf_count",
            "success_count",
            "failure_count",
            "output_dir",
            "batch_report_path",
        ]
    ]
    for engine in DEFAULT_ENGINES:
        best = best_by_engine.get(engine)
        if not best:
            rows.append([engine, "no_success", "", "", "", "", "", "", "", "", "", "", ""])
            continue
        rows.append(
            [
                engine,
                "best",
                best.get("mineru_api_max_concurrency"),
                best.get("paddle_api_max_concurrency"),
                best.get("concurrency"),
                best.get("total_elapsed_seconds"),
                best.get("page_count"),
                best.get("pages_per_second"),
                best.get("pdf_count"),
                best.get("success_count"),
                best.get("failure_count"),
                best.get("output_dir"),
                best.get("batch_report_path"),
            ]
        )
    rows.append([])
    rows.append(["failed boundary rows"])
    rows.append(
        [
            "engine",
            "mineru_api_max_concurrency",
            "paddle_api_max_concurrency",
            "concurrency",
            "status",
            "error",
            "output_dir",
        ]
    )
    for trial in trials:
        if trial.get("status") == "success":
            continue
        rows.append(
            [
                trial.get("engine"),
                trial.get("mineru_api_max_concurrency"),
                trial.get("paddle_api_max_concurrency"),
                trial.get("concurrency"),
                trial.get("status"),
                trial.get("error"),
                trial.get("output_dir"),
            ]
        )
    return rows


def build_engine_rows(trials: list[dict[str, Any]], *, engine: str) -> list[list[Any]]:
    rows = [
        [
            "engine",
            "mineru_api_max_concurrency",
            "paddle_api_max_concurrency",
            "concurrency",
            "status",
            "boundary_failure",
            "total_elapsed_seconds",
            "page_count",
            "pages_per_second",
            "pdf_count",
            "success_count",
            "failure_count",
            "skipped_count",
            "return_code",
            "output_dir",
            "batch_report_path",
            "error",
        ]
    ]
    for trial in trials:
        if trial.get("engine") != engine:
            continue
        rows.append(
            [
                trial.get("engine"),
                trial.get("mineru_api_max_concurrency"),
                trial.get("paddle_api_max_concurrency"),
                trial.get("concurrency"),
                trial.get("status"),
                trial.get("boundary_failure"),
                trial.get("total_elapsed_seconds"),
                trial.get("page_count"),
                trial.get("pages_per_second"),
                trial.get("pdf_count"),
                trial.get("success_count"),
                trial.get("failure_count"),
                trial.get("skipped_count"),
                trial.get("return_code"),
                trial.get("output_dir"),
                trial.get("batch_report_path"),
                trial.get("error"),
            ]
        )
    return rows


def build_style_map(
    rows: list[list[Any]],
    best_by_engine: dict[str, dict[str, Any]],
    sheet_name: str,
) -> dict[int, int]:
    style_by_row = {1: 1}
    if sheet_name == "Summary":
        for index, row in enumerate(rows, start=1):
            if len(row) > 1 and row[1] == "best":
                style_by_row[index] = 3
            elif len(row) > 4 and row[4] == "failed":
                style_by_row[index] = 2
        return style_by_row

    engine = sheet_name.lower()
    best = best_by_engine.get(engine)
    for index, row in enumerate(rows, start=1):
        if index == 1:
            continue
        if len(row) < 5:
            continue
        status = row[4]
        mineru_api_max = row[1]
        paddle_api_max = row[2]
        concurrency = row[3]
        if (
            best
            and mineru_api_max == best.get("mineru_api_max_concurrency")
            and paddle_api_max == best.get("paddle_api_max_concurrency")
            and concurrency == best.get("concurrency")
        ):
            style_by_row[index] = 3
        elif status != "success":
            style_by_row[index] = 2
    return style_by_row


def worksheet_xml(name: str, rows: list[list[Any]], style_by_row: dict[int, int]) -> str:
    xml_rows: list[str] = []
    for row_index, row in enumerate(rows, start=1):
        cells = []
        row_style = style_by_row.get(row_index, 0)
        for col_index, value in enumerate(row, start=1):
            ref = f"{column_letter(col_index)}{row_index}"
            cells.append(cell_xml(ref, value, row_style))
        xml_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheetViews><sheetView workbookViewId="0"/></sheetViews>'
        f'<sheetFormatPr defaultRowHeight="15"/>'
        f'<sheetData>{"".join(xml_rows)}</sheetData>'
        f'<pageMargins left="0.7" right="0.7" top="0.75" bottom="0.75" header="0.3" footer="0.3"/>'
        f'</worksheet>'
    )


def cell_xml(ref: str, value: Any, style: int) -> str:
    style_attr = f' s="{style}"' if style else ""
    if value is None:
        return f'<c r="{ref}"{style_attr}/>'
    if isinstance(value, bool):
        return f'<c r="{ref}"{style_attr} t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{ref}"{style_attr}><v>{value}</v></c>'
    text = escape(clean_xml_text(str(value)), quote=False)
    return f'<c r="{ref}"{style_attr} t="inlineStr"><is><t>{text}</t></is></c>'


def clean_xml_text(value: str) -> str:
    return "".join(
        char
        if char == "\t" or char == "\n" or char == "\r" or ord(char) >= 0x20
        else " "
        for char in value
    )


def column_letter(index: int) -> str:
    letters = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def content_types_xml(sheet_count: int) -> str:
    sheet_overrides = "".join(
        f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for index in range(1, sheet_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        f"{sheet_overrides}"
        "</Types>"
    )


def root_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )


def workbook_xml(sheet_names: list[str]) -> str:
    sheets_xml = "".join(
        f'<sheet name="{escape(name)}" sheetId="{index}" r:id="rId{index}"/>'
        for index, name in enumerate(sheet_names, start=1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{sheets_xml}</sheets>"
        "</workbook>"
    )


def workbook_rels_xml(sheet_count: int) -> str:
    rels = "".join(
        f'<Relationship Id="rId{index}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        f'Target="worksheets/sheet{index}.xml"/>'
        for index in range(1, sheet_count + 1)
    )
    rels += (
        f'<Relationship Id="rId{sheet_count + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{rels}"
        "</Relationships>"
    )


def styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="2">'
        '<font><sz val="11"/><name val="Calibri"/></font>'
        '<font><b/><sz val="11"/><name val="Calibri"/></font>'
        '</fonts>'
        '<fills count="4">'
        '<fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FFFFCCCC"/><bgColor indexed="64"/></patternFill></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FFC6EFCE"/><bgColor indexed="64"/></patternFill></fill>'
        '</fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="4">'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/>'
        '<xf numFmtId="0" fontId="0" fillId="2" borderId="0" xfId="0" applyFill="1"/>'
        '<xf numFmtId="0" fontId="1" fillId="3" borderId="0" xfId="0" applyFont="1" applyFill="1"/>'
        '</cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        '</styleSheet>'
    )


def tail_text(value: str, *, max_chars: int = 4000) -> str:
    if not value:
        return ""
    normalized = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", value)
    normalized = normalized.strip()
    return normalized[-max_chars:]


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
