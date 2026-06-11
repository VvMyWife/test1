#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from zipfile import ZipFile


CATEGORY_ORDER = ("登记文档", "手写体文档", "文书档案", "印刷体文档")
CATEGORY_ALIASES = {
    "登记": "登记文档",
    "登记文档": "登记文档",
    "手写体": "手写体文档",
    "手写体文档": "手写体文档",
    "文书": "文书档案",
    "文书档案": "文书档案",
    "增加文书": "文书档案",
    "印刷体": "印刷体文档",
    "印刷体文档": "印刷体文档",
}
XLSX_NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


@dataclass
class TestPoint:
    sheet: str
    excel_row: int
    category: str
    file_name: str
    test_point_name: str
    expected_text: str


@dataclass
class CheckResult:
    sheet: str
    excel_row: int
    category: str
    file_name: str
    test_point_name: str
    expected_text: str
    expected_length: int
    passed: bool
    status: str
    reason: str
    md_paths: str
    match_position: int | None


def main() -> int:
    args = parse_args()
    ocr_dir = Path(args.ocr_dir).expanduser().resolve()
    xlsx_path = Path(args.xlsx).expanduser().resolve() if args.xlsx else find_default_xlsx(ocr_dir)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else ocr_dir / "test_point_check"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ocr_dir.is_dir():
        raise SystemExit(f"OCR output directory does not exist: {ocr_dir}")
    if not xlsx_path.is_file():
        raise SystemExit(f"Test point workbook does not exist: {xlsx_path}")

    workbook_rows = read_xlsx_workbook(xlsx_path)
    test_points = extract_test_points(workbook_rows, sheet_name=args.sheet)
    if not test_points:
        raise SystemExit(
            "No test points found. Expected columns like 文件夹, 文件名, 测试点1, 测试点2..."
        )

    md_index = build_md_index(ocr_dir)
    results = check_test_points(
        test_points,
        md_index=md_index,
        ignore_punctuation=args.ignore_punctuation,
    )
    summary = build_summary(results)

    detail_csv = output_dir / "check_detail.csv"
    summary_csv = output_dir / "check_summary.csv"
    report_json = output_dir / "check_report.json"
    report_md = output_dir / "check_report.md"
    write_detail_csv(detail_csv, results)
    write_summary_csv(summary_csv, summary)
    write_report_json(
        report_json,
        xlsx_path=xlsx_path,
        ocr_dir=ocr_dir,
        results=results,
        summary=summary,
    )
    write_report_md(
        report_md,
        xlsx_path=xlsx_path,
        ocr_dir=ocr_dir,
        results=results,
        summary=summary,
    )

    print_summary(summary)
    print(f"DETAIL_CSV={detail_csv}")
    print(f"SUMMARY_CSV={summary_csv}")
    print(f"REPORT_JSON={report_json}")
    print(f"REPORT_MD={report_md}")

    if args.fail_on_error and summary["overall"]["failed_points"] > 0:
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether each Excel test-point text appears completely in the matching OCR Markdown."
        )
    )
    parser.add_argument(
        "--ocr-dir",
        default="output/ocr_first",
        help="OCR output directory containing category folders and Markdown files.",
    )
    parser.add_argument(
        "--xlsx",
        default=None,
        help="Test-point workbook. Defaults to the first .xlsx under --ocr-dir.",
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="Optional sheet name. Defaults to the first sheet with test-point headers.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Report output directory. Defaults to <ocr-dir>/test_point_check.",
    )
    parser.add_argument(
        "--ignore-punctuation",
        action="store_true",
        help="Ignore punctuation while matching. Whitespace is always ignored.",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit with code 1 when any test point fails.",
    )
    return parser.parse_args()


def find_default_xlsx(ocr_dir: Path) -> Path:
    candidates = sorted(
        path
        for path in ocr_dir.glob("*.xlsx")
        if not path.name.startswith("~$")
    )
    if not candidates:
        raise SystemExit(f"No .xlsx workbook found under {ocr_dir}")
    preferred = [path for path in candidates if "测试点" in path.name]
    return preferred[0] if preferred else candidates[0]


def read_xlsx_workbook(path: Path) -> dict[str, list[list[str]]]:
    with ZipFile(path) as archive:
        shared_strings = read_shared_strings(archive)
        workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
        rels_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rid_to_target = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels_root
        }

        sheets: dict[str, list[list[str]]] = {}
        for sheet_node in workbook_root.findall("a:sheets/a:sheet", XLSX_NS):
            name = sheet_node.attrib["name"]
            rid = sheet_node.attrib[
                "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
            ]
            target = rid_to_target[rid].lstrip("/")
            xml_path = target if target.startswith("xl/") else f"xl/{target}"
            sheets[name] = read_sheet_rows(archive, xml_path, shared_strings)
        return sheets


def read_shared_strings(archive: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    values: list[str] = []
    for item in root.findall("a:si", XLSX_NS):
        values.append("".join(node.text or "" for node in item.findall(".//a:t", XLSX_NS)))
    return values


def read_sheet_rows(
    archive: ZipFile,
    xml_path: str,
    shared_strings: list[str],
) -> list[list[str]]:
    root = ET.fromstring(archive.read(xml_path))
    rows: list[list[str]] = []
    for row_node in root.findall("a:sheetData/a:row", XLSX_NS):
        cells: list[tuple[int, str]] = []
        for cell_node in row_node.findall("a:c", XLSX_NS):
            ref = cell_node.attrib.get("r", "A1")
            cells.append((column_index_from_cell_ref(ref), read_cell_value(cell_node, shared_strings)))
        if not cells:
            continue
        row = [""] * (max(index for index, _ in cells) + 1)
        for index, value in cells:
            row[index] = value
        rows.append(row)
    return rows


def column_index_from_cell_ref(ref: str) -> int:
    letters = "".join(char for char in ref if char.isalpha())
    index = 0
    for char in letters:
        index = index * 26 + ord(char.upper()) - ord("A") + 1
    return max(index - 1, 0)


def read_cell_value(cell_node: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell_node.attrib.get("t")
    value_node = cell_node.find("a:v", XLSX_NS)
    if cell_type == "s" and value_node is not None and value_node.text is not None:
        index = int(value_node.text)
        return shared_strings[index] if index < len(shared_strings) else ""
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell_node.findall(".//a:t", XLSX_NS)).strip()
    if value_node is not None and value_node.text is not None:
        return value_node.text.strip()
    return ""


def extract_test_points(
    workbook_rows: dict[str, list[list[str]]],
    *,
    sheet_name: str | None,
) -> list[TestPoint]:
    sheets = {sheet_name: workbook_rows.get(sheet_name, [])} if sheet_name else workbook_rows
    points: list[TestPoint] = []
    for sheet, rows in sheets.items():
        if not rows:
            continue
        header_row_index, header = find_header_row(rows)
        if header_row_index is None:
            continue
        columns = resolve_columns(header)
        if columns is None:
            continue
        category_col, file_col, test_cols = columns
        last_category = ""
        for offset, row in enumerate(rows[header_row_index + 1 :], start=header_row_index + 2):
            category = normalize_category(get_cell(row, category_col).strip() or last_category)
            if category:
                last_category = category
            file_name = normalize_file_name(get_cell(row, file_col))
            if not category and not file_name:
                continue
            if not file_name:
                continue
            for col in test_cols:
                expected_text = get_cell(row, col).strip()
                if not expected_text:
                    continue
                points.append(
                    TestPoint(
                        sheet=sheet,
                        excel_row=offset,
                        category=category.strip(),
                        file_name=file_name,
                        test_point_name=header[col].strip() or f"测试点{col + 1}",
                        expected_text=expected_text,
                    )
                )
    return points


def find_header_row(rows: list[list[str]]) -> tuple[int | None, list[str]]:
    for index, row in enumerate(rows[:30]):
        normalized = [cell.strip() for cell in row]
        joined = "|".join(normalized)
        if "文件名" in joined and "测试点" in joined:
            return index, normalized
    return None, []


def resolve_columns(header: list[str]) -> tuple[int, int, list[int]] | None:
    category_col = find_first_header(header, {"文件夹", "分类", "类别", "文档类型"})
    file_col = find_first_header(header, {"文件名", "文件名称", "文件编号", "文档名", "编号"})
    test_cols = [
        index
        for index, value in enumerate(header)
        if value.strip().startswith("测试点") or "测试点" in value.strip()
    ]
    if category_col is None or file_col is None or not test_cols:
        return None
    return category_col, file_col, test_cols


def find_first_header(header: list[str], names: set[str]) -> int | None:
    for index, value in enumerate(header):
        if value.strip() in names:
            return index
    return None


def get_cell(row: list[str], index: int) -> str:
    return row[index] if index < len(row) else ""


def build_md_index(ocr_dir: Path) -> dict[tuple[str, str], list[Path]]:
    index: dict[tuple[str, str], list[Path]] = {}
    for md_path in sorted(ocr_dir.rglob("*.md")):
        try:
            relative = md_path.relative_to(ocr_dir)
        except ValueError:
            continue
        category = detect_category(relative)
        if not category:
            continue
        for doc_id in doc_id_candidates_from_path(relative):
            index.setdefault((category, doc_id), []).append(md_path)
    return index


def detect_category(relative_path: Path) -> str | None:
    for part in relative_path.parts:
        if part in CATEGORY_ORDER:
            return part
        normalized = normalize_category(part)
        if normalized in CATEGORY_ORDER:
            return normalized
    return normalize_category(relative_path.parts[0]) if relative_path.parts else None


def normalize_category(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "").strip()
    normalized = re.sub(r"\s+", "", normalized)
    return CATEGORY_ALIASES.get(normalized, normalized)


def doc_id_candidates_from_path(relative_path: Path) -> set[str]:
    candidates: set[str] = set()
    for part in relative_path.parts:
        stem = Path(part).stem
        candidates.update(doc_id_candidates(stem))
    return candidates


def doc_id_candidates(value: str) -> set[str]:
    raw = normalize_file_name(value)
    if not raw:
        return set()
    stripped = re.sub(r"(?i)(\.converted|_converted|-converted)$", "", raw)
    stripped = re.sub(r"(?i)\.(pdf|jpg|jpeg|png|bmp|tif|tiff|md|json)$", "", stripped)
    candidates = {raw, stripped}
    if stripped.isdigit():
        candidates.add(str(int(stripped)))
        candidates.add(stripped.zfill(4))
        candidates.add(stripped.zfill(3))
    return {item for item in candidates if item}


def normalize_file_name(value: str) -> str:
    value = str(value or "").strip()
    if re.fullmatch(r"\d+\.0", value):
        value = value[:-2]
    return value


def check_test_points(
    test_points: list[TestPoint],
    *,
    md_index: dict[tuple[str, str], list[Path]],
    ignore_punctuation: bool,
) -> list[CheckResult]:
    md_cache: dict[Path, str] = {}
    results: list[CheckResult] = []
    for point in test_points:
        md_paths: list[Path] = []
        for doc_id in doc_id_candidates(point.file_name):
            md_paths.extend(md_index.get((point.category, doc_id), []))
        md_paths = sorted(set(md_paths))
        expected_normalized = normalize_for_match(
            point.expected_text,
            ignore_punctuation=ignore_punctuation,
        )

        if not md_paths:
            results.append(
                build_result(
                    point,
                    passed=False,
                    reason="md_not_found",
                    md_paths=[],
                    match_position=None,
                    expected_normalized=expected_normalized,
                )
            )
            continue

        combined = "\n".join(read_text_cached(path, md_cache) for path in md_paths)
        combined_normalized = normalize_for_match(
            combined,
            ignore_punctuation=ignore_punctuation,
        )
        match_position = combined_normalized.find(expected_normalized)
        passed = bool(expected_normalized) and match_position >= 0
        results.append(
            build_result(
                point,
                passed=passed,
                reason="matched" if passed else "text_not_found",
                md_paths=md_paths,
                match_position=match_position if passed else None,
                expected_normalized=expected_normalized,
            )
        )
    return results


def read_text_cached(path: Path, cache: dict[Path, str]) -> str:
    if path not in cache:
        cache[path] = path.read_text(encoding="utf-8", errors="ignore")
    return cache[path]


def normalize_for_match(value: str, *, ignore_punctuation: bool) -> str:
    normalized = unicodedata.normalize("NFKC", value or "")
    normalized = re.sub(r"\s+", "", normalized)
    if ignore_punctuation:
        normalized = "".join(
            char
            for char in normalized
            if not unicodedata.category(char).startswith("P")
        )
    return normalized.casefold()


def build_result(
    point: TestPoint,
    *,
    passed: bool,
    reason: str,
    md_paths: list[Path],
    match_position: int | None,
    expected_normalized: str,
) -> CheckResult:
    return CheckResult(
        sheet=point.sheet,
        excel_row=point.excel_row,
        category=point.category,
        file_name=point.file_name,
        test_point_name=point.test_point_name,
        expected_text=point.expected_text,
        expected_length=len(expected_normalized),
        passed=passed,
        status="正确" if passed else "错误",
        reason=reason,
        md_paths=";".join(str(path) for path in md_paths),
        match_position=match_position,
    )


def build_summary(results: list[CheckResult]) -> dict[str, object]:
    by_category: dict[str, dict[str, object]] = {
        category: empty_summary_row(category)
        for category in CATEGORY_ORDER
    }
    by_document: dict[tuple[str, str], dict[str, object]] = {}

    for result in results:
        category_row = by_category.setdefault(result.category, empty_summary_row(result.category))
        update_summary_row(category_row, result)

        doc_key = (result.category, result.file_name)
        doc_row = by_document.setdefault(
            doc_key,
            {
                "category": result.category,
                "file_name": result.file_name,
                "total_points": 0,
                "passed_points": 0,
                "failed_points": 0,
                "missing_md_points": 0,
                "accuracy": 0.0,
                "all_passed": False,
            },
        )
        update_summary_row(doc_row, result)

    for row in by_category.values():
        finalize_summary_row(row)
    for row in by_document.values():
        finalize_summary_row(row)
        row["all_passed"] = row["failed_points"] == 0 and row["total_points"] > 0

    overall = empty_summary_row("OVERALL")
    for result in results:
        update_summary_row(overall, result)
    finalize_summary_row(overall)

    ordered_category_rows = [
        by_category[category]
        for category in CATEGORY_ORDER
        if by_category.get(category, {}).get("total_points", 0)
    ]
    extra_categories = [
        row
        for category, row in sorted(by_category.items())
        if category not in CATEGORY_ORDER and row.get("total_points", 0)
    ]
    return {
        "overall": overall,
        "by_category": [*ordered_category_rows, *extra_categories],
        "by_document": sorted(
            by_document.values(),
            key=lambda item: (str(item["category"]), str(item["file_name"])),
        ),
    }


def empty_summary_row(category: str) -> dict[str, object]:
    return {
        "category": category,
        "total_points": 0,
        "passed_points": 0,
        "failed_points": 0,
        "missing_md_points": 0,
        "accuracy": 0.0,
    }


def update_summary_row(row: dict[str, object], result: CheckResult) -> None:
    row["total_points"] = int(row["total_points"]) + 1
    if result.passed:
        row["passed_points"] = int(row["passed_points"]) + 1
    else:
        row["failed_points"] = int(row["failed_points"]) + 1
    if result.reason == "md_not_found":
        row["missing_md_points"] = int(row["missing_md_points"]) + 1


def finalize_summary_row(row: dict[str, object]) -> None:
    total = int(row["total_points"])
    passed = int(row["passed_points"])
    row["accuracy"] = round(passed / total, 4) if total else 0.0


def write_detail_csv(path: Path, results: list[CheckResult]) -> None:
    fields = list(asdict(results[0]).keys()) if results else [field.name for field in CheckResult.__dataclass_fields__.values()]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def write_summary_csv(path: Path, summary: dict[str, object]) -> None:
    rows = [summary["overall"], *summary["by_category"], *summary["by_document"]]
    fields = [
        "category",
        "file_name",
        "total_points",
        "passed_points",
        "failed_points",
        "missing_md_points",
        "accuracy",
        "all_passed",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_report_json(
    path: Path,
    *,
    xlsx_path: Path,
    ocr_dir: Path,
    results: list[CheckResult],
    summary: dict[str, object],
) -> None:
    payload = {
        "xlsx_path": str(xlsx_path),
        "ocr_dir": str(ocr_dir),
        "summary": summary,
        "details": [asdict(result) for result in results],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report_md(
    path: Path,
    *,
    xlsx_path: Path,
    ocr_dir: Path,
    results: list[CheckResult],
    summary: dict[str, object],
) -> None:
    failed = [result for result in results if not result.passed]
    lines = [
        "# OCR Markdown Test Point Check",
        "",
        f"- OCR dir: `{ocr_dir}`",
        f"- Test workbook: `{xlsx_path}`",
        "",
        "## Summary",
        "",
        "| Category | Total | Passed | Failed | Missing MD | Accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in [summary["overall"], *summary["by_category"]]:
        lines.append(
            "| {category} | {total_points} | {passed_points} | {failed_points} | "
            "{missing_md_points} | {accuracy:.2%} |".format(**row)
        )
    lines.extend(["", "## Failed Test Points", ""])
    if not failed:
        lines.append("No failed test points.")
    else:
        lines.append("| Category | File | Excel Row | Test Point | Reason | Expected Text |")
        lines.append("|---|---|---:|---|---|---|")
        for result in failed:
            lines.append(
                "| {} | {} | {} | {} | {} | {} |".format(
                    escape_md(result.category),
                    escape_md(result.file_name),
                    result.excel_row,
                    escape_md(result.test_point_name),
                    escape_md(result.reason),
                    escape_md(result.expected_text[:120]),
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def escape_md(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def print_summary(summary: dict[str, object]) -> None:
    print("category,total,passed,failed,missing_md,accuracy")
    for row in [summary["overall"], *summary["by_category"]]:
        print(
            "{category},{total_points},{passed_points},{failed_points},"
            "{missing_md_points},{accuracy:.2%}".format(**row)
        )


if __name__ == "__main__":
    raise SystemExit(main())
