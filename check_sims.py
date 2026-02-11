#!/usr/bin/env python3
"""
Check whether PFM simulations in a results directory completed correctly.

This script scans all immediate subfolders of a results directory (each subfolder
represents a simulation run) and verifies:
  - Required output files exist and are non-empty
  - The simulation reached the expected final time/load (based on the last value
    recorded in force text files and XDMF time stamps)
  - The final phase-field PNG (e.g. p_5000.png) exists (strict mode)
  - (Optional) Associates each failed run with its scheduler stdout log in
    pfm_dataset/logs/t_a/ by matching the line:
      seed: <seed>, out_file = pfm_dataset/results/<prefix>/<seed>

Defaults are chosen for:
  pfm_dataset/results/tension_amor_3c
which typically uses num_steps=5000 and delta_T=1e-6 => final time 0.005.

Example:
  python3 pfm_dataset/check_sims.py /path/to/pfm_dataset/results/tension_amor_3c
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_RESULTS_DIR = (
    "/projectnb/lejlab2/erfan/pfmv2/pfm_lite/pfm_dataset/results/tension_amor_3c"
)

DEFAULT_LOGS_DIR = str(Path(__file__).resolve().parent / "logs" / "t_a")

_TIME_RE = re.compile(r'<Time\s+Value="([^"]+)"')
_SEED_OUTFILE_RE = re.compile(
    r"seed:\s*(?P<seed>\d+)\s*,\s*out_file\s*=\s*(?P<out_file>\S+)"
)
_SGE_LOG_NAME_RE = re.compile(r"\.o(?P<job_id>\d+)\.(?P<task_id>\d+)$")

# Keep this conservative to avoid false positives like "error_total".
_LOG_ERROR_RE = re.compile(
    r"(Traceback\s*\(most\s+recent\s+call\s+last\)\s*:|RuntimeError|PETSC\s+ERROR|"
    r"Segmentation\s+fault|SIGSEGV|SIGKILL|MPI_Abort|Floating\s+point\s+exception|"
    r"\bKilled\b|\bkilled\b)",
    re.IGNORECASE,
)

_LOG_COMPLETED_MARKERS = (
    "Simulation completed in",
    "rank = 0 done.",
)


@dataclass(frozen=True)
class RunCheckResult:
    run_id: str
    status: str  # "PASS" | "FAIL"
    last_time_force_bot: float | None
    last_time_force_left: float | None
    last_time_p_xdmf: float | None
    last_time_u_xdmf: float | None
    has_final_png: bool
    missing_files: tuple[str, ...]
    empty_files: tuple[str, ...]
    reasons: tuple[str, ...]

@dataclass(frozen=True)
class LogInfo:
    log_file: str | None
    log_completed: bool | None
    log_error_hint: str | None


@dataclass(frozen=True)
class FailedJobRow:
    seed: str
    task_id: int | None
    job_id: int | None
    log_file: str | None


def _is_close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def _read_last_nonempty_line(path: Path, *, block_size: int = 8192, max_read: int = 1024 * 1024) -> str | None:
    """
    Return the last non-empty line in a text file.

    Reads from the end of the file in blocks (up to max_read bytes) to avoid
    loading large files into memory.
    """
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        end_pos = f.tell()
        if end_pos == 0:
            return None

        read_so_far = 0
        buffer = b""

        while end_pos > 0 and read_so_far < max_read:
            step = min(block_size, end_pos)
            end_pos -= step
            f.seek(end_pos)
            chunk = f.read(step)
            buffer = chunk + buffer
            read_so_far += step

            # Search for a non-empty line starting from the end.
            for raw in reversed(buffer.splitlines()):
                if raw.strip():
                    return raw.decode("utf-8", errors="replace")

        # If we get here, the file is either all whitespace or max_read was too small.
        # Fall back to checking if there is *any* non-empty content in what we read.
        for raw in reversed(buffer.splitlines()):
            if raw.strip():
                return raw.decode("utf-8", errors="replace")
    return None


def _parse_last_time_from_force_txt(path: Path) -> float:
    """
    Parse the last time/displacement value from a force text file.

    Expected format per line:
      <force_value> <time_value>
    """
    line = _read_last_nonempty_line(path)
    if line is None:
        raise ValueError("force file is empty (no non-empty lines)")

    parts = line.strip().split()
    if len(parts) < 2:
        raise ValueError(f"could not parse force file last line: {line!r}")

    # time is the second column
    return float(parts[1])


def _parse_last_time_from_xdmf(path: Path) -> float:
    """
    Parse the last <Time Value=\"...\"/> value from an XDMF file.
    """
    last: float | None = None
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _TIME_RE.search(line)
            if not m:
                continue
            last = float(m.group(1))

    if last is None:
        raise ValueError("no <Time Value=...> tags found")
    return last


def _read_tail_text(path: Path, *, max_bytes: int = 200_000) -> str:
    """
    Read and decode the tail of a text file.

    This is used to heuristically detect errors/completion in large scheduler logs
    without scanning the entire file.
    """
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - max_bytes))
        data = f.read()
    return data.decode("utf-8", errors="replace")


def _summarize_log(path: Path) -> tuple[bool | None, str | None]:
    """
    Return (log_completed, log_error_hint) by inspecting the tail of a log file.

    - log_completed: True if completion markers are found in the tail
    - log_error_hint: last line in the tail that matches a conservative error regex
    """
    try:
        tail = _read_tail_text(path)
    except OSError:
        return None, None

    completed = any(marker in tail for marker in _LOG_COMPLETED_MARKERS)

    last_err: str | None = None
    for line in tail.splitlines():
        if _LOG_ERROR_RE.search(line):
            last_err = line.strip()

    if last_err is not None and len(last_err) > 500:
        last_err = last_err[:500] + "..."

    return completed, last_err


def _parse_sge_job_and_task_id_from_log_file(path: Path) -> tuple[int | None, int | None]:
    """
    Parse (job_id, task_id) from an SGE array stdout log filename.

    Example:
      t_a.o2951272.5 -> (2951272, 5)
    """
    m = _SGE_LOG_NAME_RE.search(path.name)
    if not m:
        return None, None
    try:
        return int(m.group("job_id")), int(m.group("task_id"))
    except ValueError:
        return None, None


def _file_missing_or_empty(run_dir: Path, relpath: str) -> tuple[bool, bool]:
    """
    Returns (missing, empty) for a file relative to run_dir.
    """
    p = run_dir / relpath
    if not p.exists():
        return True, False
    try:
        if p.is_file() and p.stat().st_size == 0:
            return False, True
    except OSError:
        # Treat stat failures as empty to force a FAIL, but keep scanning.
        return False, True
    return False, False


def _list_run_dirs(results_dir: Path) -> list[Path]:
    run_dirs: list[Path] = []
    with os.scandir(results_dir) as it:
        for entry in it:
            if entry.is_dir():
                run_dirs.append(Path(entry.path))

    def sort_key(p: Path):
        try:
            return (0, int(p.name))
        except ValueError:
            return (1, p.name)

    run_dirs.sort(key=sort_key)
    return run_dirs


def _index_logs(
    logs_dir: Path,
    *,
    expected_out_file_prefix: str,
    target_run_ids: set[str],
    max_lines: int = 500,
) -> dict[str, list[Path]]:
    """
    Build mapping: run_id(seed as string) -> list of log files.

    We scan only the first `max_lines` lines of each log file to find the
    "seed: ..., out_file = ..." line and filter it by the expected prefix.
    """
    seed_to_logs: dict[str, list[Path]] = {}
    if not logs_dir.exists() or not logs_dir.is_dir():
        return seed_to_logs

    with os.scandir(logs_dir) as it:
        for entry in it:
            if not entry.is_file():
                continue
            p = Path(entry.path)
            try:
                with p.open("r", encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f):
                        if i >= max_lines:
                            break
                        m = _SEED_OUTFILE_RE.search(line)
                        if not m:
                            continue
                        seed = m.group("seed")
                        out_file = m.group("out_file")
                        if not out_file.startswith(expected_out_file_prefix):
                            break
                        if seed not in target_run_ids:
                            break
                        seed_to_logs.setdefault(seed, []).append(p)
                        break
            except OSError:
                continue

    return seed_to_logs


def _choose_latest(paths: list[Path]) -> Path:
    def key(p: Path):
        try:
            st = p.stat()
            return (st.st_mtime, st.st_size, str(p))
        except OSError:
            return (0.0, 0, str(p))

    return max(paths, key=key)


def _check_run(
    run_dir: Path,
    *,
    expected_final_time: float,
    expected_last_step: int,
    tol: float,
) -> RunCheckResult:
    required_files = (
        "p_unit.xdmf",
        "u_unit.xdmf",
        "p_unit.h5",
        "u_unit.h5",
        "force_bot_rxn.txt",
        "force_left_rxn.txt",
        "force_disp_bot_rxn.png",
        "force_disp_left_rxn.png",
    )
    final_png = f"p_{expected_last_step}.png"

    missing: list[str] = []
    empty: list[str] = []
    reasons: list[str] = []

    for rel in required_files:
        is_missing, is_empty = _file_missing_or_empty(run_dir, rel)
        if is_missing:
            missing.append(rel)
        elif is_empty:
            empty.append(rel)

    # final PNG is part of strict completion criteria
    final_png_missing, final_png_empty = _file_missing_or_empty(run_dir, final_png)
    has_final_png = not final_png_missing and not final_png_empty
    if final_png_missing:
        reasons.append(f"missing:{final_png}")
    elif final_png_empty:
        reasons.append(f"empty:{final_png}")

    last_time_force_bot: float | None = None
    last_time_force_left: float | None = None
    last_time_p_xdmf: float | None = None
    last_time_u_xdmf: float | None = None

    # Parse times only if the source files exist and are non-empty.
    try:
        if "force_bot_rxn.txt" not in missing and "force_bot_rxn.txt" not in empty:
            last_time_force_bot = _parse_last_time_from_force_txt(run_dir / "force_bot_rxn.txt")
            if not _is_close(last_time_force_bot, expected_final_time, tol):
                reasons.append(
                    f"force_bot_last_time={last_time_force_bot:g} != {expected_final_time:g}"
                )
    except Exception as e:
        reasons.append(f"force_bot_parse_error:{type(e).__name__}:{e}")

    try:
        if "force_left_rxn.txt" not in missing and "force_left_rxn.txt" not in empty:
            last_time_force_left = _parse_last_time_from_force_txt(run_dir / "force_left_rxn.txt")
            if not _is_close(last_time_force_left, expected_final_time, tol):
                reasons.append(
                    f"force_left_last_time={last_time_force_left:g} != {expected_final_time:g}"
                )
    except Exception as e:
        reasons.append(f"force_left_parse_error:{type(e).__name__}:{e}")

    try:
        if "p_unit.xdmf" not in missing and "p_unit.xdmf" not in empty:
            last_time_p_xdmf = _parse_last_time_from_xdmf(run_dir / "p_unit.xdmf")
            if not _is_close(last_time_p_xdmf, expected_final_time, tol):
                reasons.append(
                    f"p_xdmf_last_time={last_time_p_xdmf:g} != {expected_final_time:g}"
                )
    except Exception as e:
        reasons.append(f"p_xdmf_parse_error:{type(e).__name__}:{e}")

    try:
        if "u_unit.xdmf" not in missing and "u_unit.xdmf" not in empty:
            last_time_u_xdmf = _parse_last_time_from_xdmf(run_dir / "u_unit.xdmf")
            if not _is_close(last_time_u_xdmf, expected_final_time, tol):
                reasons.append(
                    f"u_xdmf_last_time={last_time_u_xdmf:g} != {expected_final_time:g}"
                )
    except Exception as e:
        reasons.append(f"u_xdmf_parse_error:{type(e).__name__}:{e}")

    # Missing/empty required files are reasons too (kept separate in CSV, but also included here)
    for rel in missing:
        reasons.append(f"missing:{rel}")
    for rel in empty:
        reasons.append(f"empty:{rel}")

    status = "PASS" if len(reasons) == 0 else "FAIL"
    return RunCheckResult(
        run_id=run_dir.name,
        status=status,
        last_time_force_bot=last_time_force_bot,
        last_time_force_left=last_time_force_left,
        last_time_p_xdmf=last_time_p_xdmf,
        last_time_u_xdmf=last_time_u_xdmf,
        has_final_png=has_final_png,
        missing_files=tuple(missing),
        empty_files=tuple(empty),
        reasons=tuple(reasons),
    )


def _write_csv(
    path: Path, results: Iterable[RunCheckResult], *, log_info: dict[str, LogInfo] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "status",
        "last_time_force_bot",
        "last_time_force_left",
        "last_time_p_xdmf",
        "last_time_u_xdmf",
        "has_final_png",
        "log_file",
        "log_completed",
        "log_error_hint",
        "missing_files",
        "empty_files",
        "reasons",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            li = log_info.get(r.run_id) if log_info is not None else None
            w.writerow(
                {
                    "run_id": r.run_id,
                    "status": r.status,
                    "last_time_force_bot": "" if r.last_time_force_bot is None else f"{r.last_time_force_bot:.17g}",
                    "last_time_force_left": "" if r.last_time_force_left is None else f"{r.last_time_force_left:.17g}",
                    "last_time_p_xdmf": "" if r.last_time_p_xdmf is None else f"{r.last_time_p_xdmf:.17g}",
                    "last_time_u_xdmf": "" if r.last_time_u_xdmf is None else f"{r.last_time_u_xdmf:.17g}",
                    "has_final_png": "1" if r.has_final_png else "0",
                    "log_file": "" if li is None or li.log_file is None else li.log_file,
                    "log_completed": ""
                    if li is None or li.log_completed is None
                    else ("1" if li.log_completed else "0"),
                    "log_error_hint": "" if li is None or li.log_error_hint is None else li.log_error_hint,
                    "missing_files": ";".join(r.missing_files),
                    "empty_files": ";".join(r.empty_files),
                    "reasons": ";".join(r.reasons),
                }
            )


def _write_failed_jobs_csv(path: Path, rows: list[FailedJobRow]) -> None:
    """
    Write a CSV containing failed seeds and their SGE array task ids (for reruns).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["seed", "task_id", "job_id", "log_file"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "seed": r.seed,
                    "task_id": "" if r.task_id is None else str(r.task_id),
                    "job_id": "" if r.job_id is None else str(r.job_id),
                    "log_file": "" if r.log_file is None else r.log_file,
                }
            )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Scan a PFM results directory and report which simulation runs completed correctly."
        )
    )
    p.add_argument(
        "results_dir",
        nargs="?",
        default=DEFAULT_RESULTS_DIR,
        help=f"Path to results directory (default: {DEFAULT_RESULTS_DIR})",
    )
    p.add_argument(
        "--expected-final-time",
        type=float,
        default=0.005,
        help="Expected final time/displacement (default: 0.005)",
    )
    p.add_argument(
        "--expected-last-step",
        type=int,
        default=5000,
        help="Expected last step index for final PNG (default: 5000 -> p_5000.png)",
    )
    p.add_argument(
        "--tol",
        type=float,
        default=1e-8,
        help="Absolute tolerance for final time comparisons (default: 1e-8)",
    )
    p.add_argument(
        "--csv",
        default=None,
        help="CSV output path (FAIL-only; default: <results_dir>/check_sims_report.csv)",
    )
    p.add_argument(
        "--logs-dir",
        default=DEFAULT_LOGS_DIR,
        help=f"Directory containing stdout logs (default: {DEFAULT_LOGS_DIR})",
    )
    p.add_argument(
        "--failed-jobs-csv",
        default=None,
        help="Output CSV for failed SGE array tasks (default: <results_dir>/failed_jobs.csv)",
    )
    p.add_argument(
        "--max-failed-print",
        type=int,
        default=50,
        help="Maximum number of failed runs to print with reasons (default: 50)",
    )
    p.add_argument(
        "--logs-scan-lines",
        type=int,
        default=500,
        help="Max lines to scan per log file to find the seed/out_file line (default: 500)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    results_dir = Path(args.results_dir).expanduser()
    if not results_dir.exists():
        print(f"ERROR: results_dir does not exist: {results_dir}", file=sys.stderr)
        return 2
    if not results_dir.is_dir():
        print(f"ERROR: results_dir is not a directory: {results_dir}", file=sys.stderr)
        return 2

    csv_path = (
        Path(args.csv).expanduser()
        if args.csv is not None
        else (results_dir / "check_sims_report.csv")
    )
    failed_jobs_csv_path = (
        Path(args.failed_jobs_csv).expanduser()
        if args.failed_jobs_csv is not None
        else (results_dir / "failed_jobs.csv")
    )

    run_dirs = _list_run_dirs(results_dir)
    if len(run_dirs) == 0:
        print(f"No run subfolders found in: {results_dir}", file=sys.stderr)
        return 2

    results: list[RunCheckResult] = []
    for run_dir in run_dirs:
        results.append(
            _check_run(
                run_dir,
                expected_final_time=args.expected_final_time,
                expected_last_step=args.expected_last_step,
                tol=args.tol,
            )
        )

    total = len(results)
    passed = sum(1 for r in results if r.status == "PASS")
    failed_results = [r for r in results if r.status == "FAIL"]
    failed = len(failed_results)

    # Build log mapping ONLY for failed runs (as requested).
    log_info: dict[str, LogInfo] = {}
    logs_dir = Path(args.logs_dir).expanduser()
    if failed > 0 and logs_dir.exists() and logs_dir.is_dir():
        expected_out_file_prefix = f"pfm_dataset/results/{results_dir.name}/"
        target_run_ids = {r.run_id for r in failed_results}
        seed_to_logs = _index_logs(
            logs_dir,
            expected_out_file_prefix=expected_out_file_prefix,
            target_run_ids=target_run_ids,
            max_lines=args.logs_scan_lines,
        )
        for run_id, paths in seed_to_logs.items():
            chosen = _choose_latest(paths)
            completed, err_hint = _summarize_log(chosen)
            log_info[run_id] = LogInfo(
                log_file=str(chosen),
                log_completed=completed,
                log_error_hint=err_hint,
            )

    # Write CSV report containing ONLY failed runs, with log file path if available.
    _write_csv(csv_path, failed_results, log_info=log_info)

    # Write a separate CSV containing the failed SGE array task ids.
    failed_job_rows: list[FailedJobRow] = []
    for r in failed_results:
        li = log_info.get(r.run_id)
        log_file = None if li is None else li.log_file
        job_id: int | None = None
        task_id: int | None = None
        if log_file:
            job_id, task_id = _parse_sge_job_and_task_id_from_log_file(Path(log_file))
        failed_job_rows.append(
            FailedJobRow(
                seed=r.run_id,
                task_id=task_id,
                job_id=job_id,
                log_file=log_file,
            )
        )

    # Sort by task id when available (helps with qsub -t lists).
    failed_job_rows.sort(
        key=lambda x: (1_000_000_000 if x.task_id is None else x.task_id, x.seed)
    )
    _write_failed_jobs_csv(failed_jobs_csv_path, failed_job_rows)

    print(f"Results dir: {results_dir}")
    print(f"Logs dir: {Path(args.logs_dir).expanduser()}")
    print(f"Scanned runs: {total}")
    print(f"PASS: {passed}")
    print(f"FAIL: {failed}")
    print(f"CSV report (FAIL only): {csv_path}")
    print(f"Failed jobs CSV (seed + array task id): {failed_jobs_csv_path}")

    if failed > 0:
        print("")
        print(f"Failed runs (showing up to {args.max_failed_print}):")
        for r in failed_results[: args.max_failed_print]:
            reason_str = "; ".join(r.reasons) if r.reasons else "unknown"
            print(f"{r.run_id}\t{reason_str}")
        if failed > args.max_failed_print:
            print(
                f"... (+{failed - args.max_failed_print} more failed runs; see CSV report)"
            )

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

