"""Run the full female emotion voice modulation experiment (generate + analyze)."""
from __future__ import annotations

import argparse
from pathlib import Path

from analyze_clips import DEFAULT_OUTPUT_DIR, DEFAULT_REFERENCE_DIR, analyze_directory
from generate_clips import DEFAULT_TEXT, generate_clips


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate WAV clips for all female emotion tensors and analyze pitch & tone."
    )
    parser.add_argument(
        "--text",
        type=str,
        default=DEFAULT_TEXT,
        help="Prompt text to synthesize across every female emotion tensor.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for generated WAV clips and analysis reports.",
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        default=str(DEFAULT_REFERENCE_DIR),
        help="Directory containing reference *_female.wav files (default: voices/output/).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Torch RNG seed used before each synthesis.",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip synthesis and only run acoustic analysis on existing WAV files.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not args.skip_generate:
        generate_clips(output_dir=out_dir, text=args.text, seed=args.seed)
        print()
    analyze_directory(input_dir=out_dir, reference_dir=args.reference_dir)


if __name__ == "__main__":
    main()
