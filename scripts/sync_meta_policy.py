#!/usr/bin/env python3
"""Promote an evolution output's best policy to the persistent Hub champion."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.meta_policy import META_POLICY_REPO, publish_policy_if_improved


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--repo-id", default=META_POLICY_REPO)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=60.0)
    args = parser.parse_args(argv)
    seen_mtime = None
    while True:
        policy = args.output_dir / "best_policy.npz"
        metadata_path = args.output_dir / "best_policy.json"
        if policy.exists() and metadata_path.exists():
            mtime = max(policy.stat().st_mtime_ns, metadata_path.stat().st_mtime_ns)
            if mtime != seen_mtime:
                metadata = json.loads(metadata_path.read_text())
                published, previous = publish_policy_if_improved(
                    policy, metadata, repo_id=args.repo_id
                )
                print(
                    f"[meta-policy-sync] {'published' if published else 'retained'} "
                    f"candidate={metadata['validation_score']} previous={previous}",
                    flush=True,
                )
                seen_mtime = mtime
        elif not args.watch:
            raise FileNotFoundError(f"No best policy found under {args.output_dir}")
        if not args.watch:
            return 0
        time.sleep(max(args.interval, 1.0))


if __name__ == "__main__":
    raise SystemExit(main())
