"""Benchmark runner.

Three arms, so each component can be credited separately:

    no_ir     paper text -> model.py generated, no repair  (the baseline to beat)
    no_repair paper -> spec -> model.py generated          (isolates the spec IR)
    full      paper -> spec -> model.py generated + repair (isolates the repair loop)
    compiled  paper -> spec -> model.py compiled           (isolates generation itself)

The first three all generate model.py with a model, so they stay comparable
with each other; only what the generator is shown changes. `compiled` replaces
the generator with emit.py, which is how the shipped default behaves.

Usage:
    python -m runmypaper.bench gold resnet      # bootstrap a gold spec to correct
    python -m runmypaper.bench run --arm full
    python -m runmypaper.bench report
"""
import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path
from statistics import mean
from typing import List, Optional

import yaml

from . import baseline, codegen, emit, extract, ingest, metrics, verify
from .config import ROOT
from .spec import PaperSpec

BENCH = ROOT / "bench"
GOLD = BENCH / "gold"
RUNS = BENCH / "runs"
RESULTS = BENCH / "results.json"
ARMS = ("no_ir", "no_repair", "full", "compiled")


def papers() -> List[dict]:
    return yaml.safe_load((BENCH / "papers.yaml").read_text())


def gold_path(name: str) -> Path:
    return GOLD / f"{name}.json"


def measure_params(out: Path) -> Optional[int]:
    """Parameter count of the built model, from the forward gate's output."""
    p = subprocess.run([sys.executable, "smoke.py", "--stage", "forward"],
                       cwd=out, capture_output=True, text=True, timeout=300)
    if p.returncode != 0:
        return None
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])["params"]
    except (json.JSONDecodeError, KeyError, IndexError):
        return None


# --- one paper, one arm -----------------------------------------------------

def run_one(entry: dict, arm: str, no_cache: bool = False) -> dict:
    name = entry["name"]
    row = {"name": name, "id": entry["id"], "arm": arm}
    gold = PaperSpec.model_validate_json(gold_path(name).read_text())
    doc = ingest.load_source(entry["id"])
    out = RUNS / arm / name

    if arm == "no_ir":
        baseline.build(doc, gold, out, use_cache=not no_cache)
        spec_for_repair, max_repairs = gold, 0
    else:
        spec = extract.extract(doc, use_cache=not no_cache,
                               variant=entry.get("note"))
        row.update(metrics.compare_specs(spec, gold).as_dict())
        row["unresolved"] = len(spec.unresolved)
        codegen.scaffold(spec, out)
        if arm == "compiled":
            # Refusing to compile is a result, not a crash: record why and
            # leave the gates to report what that costs.
            reasons = emit.unsupported(spec)
            if reasons:
                row["error"] = f"not compilable: {'; '.join(reasons)}"
                return row
            built = codegen.write_model(spec, out, mode="emit")
        else:
            # Forced, so the three generation arms stay comparable with each
            # other even though the shipped default now compiles.
            built = codegen.write_model(spec, out, use_cache=not no_cache,
                                        mode="llm")
        row["codegen"] = built.method
        spec_for_repair = spec
        max_repairs = 3 if arm == "full" else 0

    result = verify.verify(spec_for_repair, out, max_repairs=max_repairs,
                           on_event=lambda m: None)
    row["gates"] = result.passed
    row["failed_gate"] = result.failure.stage if result.failure else None
    row["repairs"] = result.repairs
    row["params"] = measure_params(out)
    row["param_ratio"] = metrics.param_ratio(
        row["params"], gold.architecture.param_count_reported)
    return row


# --- commands ---------------------------------------------------------------

def cmd_gold(args) -> int:
    """Extract a spec as a starting point for hand labelling.

    The output is NOT gold until a human has checked it against the paper.
    """
    entry = next((e for e in papers() if e["name"] == args.name), None)
    if entry is None:
        raise SystemExit(f"{args.name} is not in bench/papers.yaml")
    dest = gold_path(args.name)
    if dest.exists() and not args.force:
        raise SystemExit(f"{dest} exists; pass --force to overwrite")
    spec = extract.extract(ingest.load_source(entry["id"]),
                           use_cache=not args.no_cache,
                           variant=entry.get("note"))
    GOLD.mkdir(parents=True, exist_ok=True)
    dest.write_text(spec.model_dump_json(indent=2))
    print(f"draft -> {dest}\nCorrect it against the paper before scoring anything.",
          file=sys.stderr)
    return 0


def cmd_run(args) -> int:
    labelled = [e for e in papers() if gold_path(e["name"]).exists()]
    skipped = [e["name"] for e in papers() if not gold_path(e["name"]).exists()]
    if not labelled:
        raise SystemExit(
            f"No gold specs in {GOLD}. Run `bench gold <name>` and correct the "
            "draft by hand first.")
    if skipped:
        print(f"no gold spec, skipping: {', '.join(skipped)}", file=sys.stderr)

    arms = [args.arm] if args.arm else list(ARMS)
    rows = json.loads(RESULTS.read_text()) if RESULTS.exists() and not args.fresh else []
    rows = [r for r in rows if r["arm"] not in arms]

    for arm in arms:
        for entry in labelled:
            print(f"{arm}/{entry['name']} ...", file=sys.stderr, flush=True)
            try:
                rows.append(run_one(entry, arm, no_cache=args.no_cache))
            except Exception as exc:
                traceback.print_exc()
                rows.append({"name": entry["name"], "id": entry["id"], "arm": arm,
                             "error": f"{type(exc).__name__}: {exc}"})

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(rows, indent=2))
    return cmd_report(args)


def _avg(rows, key):
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return round(mean(vals), 3) if vals else None


def cmd_report(args) -> int:
    if not RESULTS.exists():
        raise SystemExit(f"no results at {RESULTS}; run `bench run` first")
    rows = json.loads(RESULTS.read_text())
    stages = [s for s, _ in verify.GATES]

    print(f"\n{'arm':10s} {'n':>3s} {'spec_f1':>8s} {'arch_f1':>8s} "
          + " ".join(f"{s:>8s}" for s in stages) + f" {'repairs':>8s}")
    print("-" * (10 + 4 + 18 + 9 * len(stages) + 9))
    for arm in ARMS:
        a = [r for r in rows if r["arm"] == arm and "error" not in r]
        if not a:
            continue
        rates = [f"{sum(s in r['gates'] for r in a) / len(a):8.2f}" for s in stages]
        print(f"{arm:10s} {len(a):3d} "
              f"{str(_avg(a, 'spec_f1') or '-'):>8s} {str(_avg(a, 'arch_f1') or '-'):>8s} "
              + " ".join(rates) + f" {str(_avg(a, 'repairs')):>8s}")

    errs = [r for r in rows if "error" in r]
    if errs:
        print(f"\nerrors ({len(errs)}):")
        for r in errs:
            print(f"  {r['arm']}/{r['name']}: {r['error']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="runmypaper.bench")
    ap.add_argument("--no-cache", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gold", help="draft a gold spec for hand correction")
    g.add_argument("name")
    g.add_argument("--force", action="store_true")
    g.set_defaults(func=cmd_gold)

    r = sub.add_parser("run", help="run the benchmark")
    r.add_argument("--arm", choices=ARMS, help="default: all three")
    r.add_argument("--fresh", action="store_true", help="discard previous results")
    r.set_defaults(func=cmd_run)

    p = sub.add_parser("report", help="print the results table")
    p.set_defaults(func=cmd_report)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
