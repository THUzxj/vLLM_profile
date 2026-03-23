#!/usr/bin/env python3
import argparse
import logging
import os
import sys


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def _import_profile_merger():
    """
    Import ProfileMerger from either:
    - in-repo source checkout (preferred for this workspace), or
    - an installed 'sglang' package.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    in_repo_sglang_python = os.path.join(repo_root, "sglang", "python")
    if os.path.isdir(in_repo_sglang_python) and in_repo_sglang_python not in sys.path:
        sys.path.insert(0, in_repo_sglang_python)

    from sglang.srt.utils.profile_merger import ProfileMerger  # type: ignore

    return ProfileMerger


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge Chrome trace files for a given profile_id."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory containing per-rank *.trace.json.gz files.",
    )
    parser.add_argument(
        "--profile-id",
        required=True,
        help="Profile id prefix used to discover trace files.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="If set, only merge files whose basename starts with this prefix.",
    )
    parser.add_argument(
        "--suffix",
        default=None,
        help="If set, only merge files whose basename ends with this suffix.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging level (default: INFO). Use DEBUG to see discovery details.",
    )
    args = parser.parse_args()

    _setup_logging(args.log_level)

    ProfileMerger = _import_profile_merger()
    merger = ProfileMerger(
        output_dir=args.output_dir,
        profile_id=args.profile_id,
        prefix=args.prefix,
        suffix=args.suffix,
    )
    merged_path = merger.merge_chrome_traces()
    print(merged_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

