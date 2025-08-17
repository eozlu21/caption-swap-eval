from __future__ import annotations

import argparse
import os
from dotenv import load_dotenv

from state import initialize_state
from engine import run_until_solved


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Iterative caption–cartoon swap loop")
    parser.add_argument("--n", type=int, required=True, help="Number of cartoons/captions to sample (>=2)")
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "42")), help="RNG seed")
    parser.add_argument("--run-id", type=str, default=None, help="Optional run id; defaults to timestamp-based")
    parser.add_argument("--out", type=str, default="runs", help="Output root directory")
    parser.add_argument("--max-turns", type=int, default=100, help="Safety cap on iterations")

    args = parser.parse_args()

    if args.n < 2:
        raise SystemExit("--n must be >= 2 to create a derangement")

    full, _ = initialize_state(n=args.n, seed=args.seed, run_id=args.run_id)

    out_dir = run_until_solved(full, out_root=args.out, max_turns=args.max_turns)
    print(out_dir)


if __name__ == "__main__":
    main()
