"""End-to-end pipeline: paper -> spec -> code -> verified run.

    python -m runmypaper.cli run 1512.03385 -o out/resnet     # arXiv id
    python -m runmypaper.cli run paper.pdf -o out/resnet      # local PDF
    python -m runmypaper.cli extract 1512.03385               # stop at the spec
    python -m runmypaper.cli build out/resnet/spec.json -o out/resnet
"""
import argparse
import json
import sys
from pathlib import Path

from . import codegen, extract, ingest, verify
from .spec import PaperSpec

VARIANT_HELP = ('which configuration to extract, e.g. "ResNet-20, CIFAR-10". '
                "Papers report several; naming one keeps the spec comparable "
                "with a gold spec. Default: the smallest fully specified.")


def _report(spec: PaperSpec, result, out: Path) -> dict:
    low = [e.field for e in spec.evidence if e.confidence < 0.6]
    return {
        "title": spec.title,
        "output": str(out),
        "gates_passed": result.passed if result else [],
        "failed_gate": result.failure.stage if result and result.failure else None,
        "repairs": result.repairs if result else 0,
        "unresolved": spec.unresolved,
        "low_confidence": low,
    }


def cmd_check(args) -> int:
    """Confirm the provider is reachable before spending any quota."""
    from . import config, llm

    for k, v in config.summary().items():
        print(f"  {k:10s} {v}")

    if config.PROVIDER == "openai_compat":
        try:
            models = llm.list_models()
        except llm.LLMError as exc:
            print(f"\ncannot list models: {exc}", file=sys.stderr)
            return 1
        # Endpoints list ids with a "models/" prefix but accept the bare name.
        bare = {m.split("/", 1)[-1] for m in models}
        chosen = config.MODEL_ID.split("/", 1)[-1]
        print(f"\n{len(models)} models offered:")
        for m in models:
            print(f"  {'* ' if m.split('/', 1)[-1] == chosen else '  '}{m}")
        if chosen and chosen not in bare:
            print(f"\nRMP_MODEL={config.MODEL_ID} is not in that list.",
                  file=sys.stderr)
            return 1
        if not config.MODEL_ID:
            print("\nRMP_MODEL is unset; pick one from the list above.",
                  file=sys.stderr)
            return 1

    print("\nsending one request ...")
    try:
        reply = llm.generate("Reply with exactly: OK", "ping", use_cache=False)
    except (llm.LLMError, llm.LLMRefusal, SystemExit) as exc:
        print(f"failed: {exc}", file=sys.stderr)
        return 1
    print(f"reply: {reply.strip()[:200]!r}\nprovider is working.")
    return 0


def cmd_extract(args) -> int:
    doc = ingest.load_source(args.paper)
    print(f"{len(doc.sections)} sections: "
          f"{', '.join(s.name for s in doc.sections[:8])}", file=sys.stderr)
    spec = extract.extract(doc, use_cache=not args.no_cache,
                           variant=args.variant)
    dest = Path(args.out or "spec.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(spec.model_dump_json(indent=2))
    print(f"spec -> {dest}", file=sys.stderr)
    return 0


def _build(spec: PaperSpec, out: Path, no_cache: bool, skip_verify: bool) -> int:
    codegen.scaffold(spec, out)
    codegen.write_model(spec, out, use_cache=not no_cache)
    print(f"generated {out}", file=sys.stderr)
    result = None if skip_verify else verify.verify(
        spec, out, on_event=lambda m: print(m, file=sys.stderr))
    report = _report(spec, result, out)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if (result is None or result.ok) else 1


def cmd_build(args) -> int:
    spec = PaperSpec.model_validate_json(Path(args.spec).read_text())
    return _build(spec, Path(args.out), args.no_cache, args.skip_verify)


def cmd_run(args) -> int:
    doc = ingest.load_source(args.paper)
    spec = extract.extract(doc, use_cache=not args.no_cache,
                           variant=args.variant)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    return _build(spec, out, args.no_cache, args.skip_verify)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="runmypaper")
    ap.add_argument("--no-cache", action="store_true", help="bypass the LLM cache")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="verify the provider and model work")
    c.set_defaults(func=cmd_check)

    e = sub.add_parser("extract", help="paper -> spec.json")
    e.add_argument("paper", help="arXiv id/URL or a local PDF")
    e.add_argument("-o", "--out")
    e.add_argument("--variant", help=VARIANT_HELP)
    e.set_defaults(func=cmd_extract)

    b = sub.add_parser("build", help="spec.json -> verified project")
    b.add_argument("spec")
    b.add_argument("-o", "--out", required=True)
    b.add_argument("--skip-verify", action="store_true")
    b.set_defaults(func=cmd_build)

    r = sub.add_parser("run", help="paper -> verified project")
    r.add_argument("paper", help="arXiv id/URL or a local PDF")
    r.add_argument("-o", "--out", required=True)
    r.add_argument("--variant", help=VARIANT_HELP)
    r.add_argument("--skip-verify", action="store_true")
    r.set_defaults(func=cmd_run)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
