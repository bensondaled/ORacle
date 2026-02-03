#!/usr/bin/env python3
"""
Generate Preoperative Case Embeddings
=====================================

CLI script to generate preoperative case embeddings from DuckDB databases.

Usage:
    # Full run (all cases)
    python generate_preop_embeddings.py \
        --caseinfo-db /path/to/caseinfo.duckdb \
        --meds-db /path/to/meds.duckdb \
        --output embeddings/preop_embeddings.parquet

    # Debug mode (1% sample)
    python generate_preop_embeddings.py --debug

    # Custom debug fraction
    python generate_preop_embeddings.py --debug --debug-frac 0.05

    # Specific case IDs from file
    python generate_preop_embeddings.py --case-ids cases.txt

    # With custom batch size
    python generate_preop_embeddings.py --batch-size 5000

Examples:
    # Quick test with 100 cases
    python generate_preop_embeddings.py --debug --debug-frac 0.001

    # Full production run
    python generate_preop_embeddings.py \
        --output /nfs/turbo/.../preop_embeddings.parquet \
        --batch-size 10000
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from preop_case_embeddings import (
    EmbeddingConfig,
    PreopCaseEmbedder,
    load_embeddings,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Default paths
DEFAULT_CASEINFO_DB = "/nfs/turbo/umms-sachinkh/PCRC 247 Aghaeepour/case_info/pcrc_caseinfo.duckdb"
DEFAULT_MEDS_DB = "/nfs/turbo/umms-sachinkh/PCRC 247 Aghaeepour/pcrc_0247_preop_meds.duckdb"
DEFAULT_OUTPUT = "embeddings/preop_embeddings.parquet"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate preoperative case embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input/output
    parser.add_argument(
        "--caseinfo-db",
        type=str,
        default=DEFAULT_CASEINFO_DB,
        help=f"Path to case info DuckDB database (default: {DEFAULT_CASEINFO_DB})",
    )
    parser.add_argument(
        "--meds-db",
        type=str,
        default=DEFAULT_MEDS_DB,
        help=f"Path to medications DuckDB database (default: {DEFAULT_MEDS_DB})",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output parquet file path (default: {DEFAULT_OUTPUT})",
    )

    # Processing options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Number of cases per processing batch (default: 10000)",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=256,
        help="Number of texts per embedding batch (default: 256)",
    )
    parser.add_argument(
        "--pca-fit-batches",
        type=int,
        default=10,
        help="Number of batches to fit PCA on (default: 10 = 100k cases)",
    )

    # Debug/subset options
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: process only 1%% of cases",
    )
    parser.add_argument(
        "--debug-frac",
        type=float,
        default=0.01,
        help="Fraction of cases to process in debug mode (default: 0.01)",
    )
    parser.add_argument(
        "--case-ids",
        type=str,
        help="File with specific case IDs to process (one per line)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of cases to process",
    )

    # Model options
    parser.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-small-en-v1.5",
        choices=[
            "BAAI/bge-small-en-v1.5",
            "BAAI/bge-base-en-v1.5",
            "sentence-transformers/all-MiniLM-L6-v2",
        ],
        help="Embedding model (default: BAAI/bge-small-en-v1.5)",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable FP16 precision (uses FP32)",
    )

    # Dimension options
    parser.add_argument(
        "--proc-dim",
        type=int,
        default=384,
        help="Procedure embedding dimension (default: 384)",
    )
    parser.add_argument(
        "--med-dim",
        type=int,
        default=64,
        help="Medication embedding dimension (default: 64)",
    )
    parser.add_argument(
        "--struct-dim",
        type=int,
        default=96,
        help="Structured feature embedding dimension (default: 96)",
    )

    # Misc
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run verification checks on output",
    )

    return parser.parse_args()


def load_case_ids_from_file(path: str) -> list:
    """Load case IDs from a text file (one per line)."""
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def verify_embeddings(path: str) -> bool:
    """
    Run verification checks on generated embeddings.

    Returns True if all checks pass.
    """
    logger.info("Running verification checks...")

    df, X = load_embeddings(path)

    checks_passed = True

    # Check 1: Shape
    n_cases, n_dims = X.shape
    logger.info(f"  Shape: {X.shape}")
    if n_dims != 544:
        logger.warning(f"  Expected 544 dimensions, got {n_dims}")
        checks_passed = False
    else:
        logger.info("  Dimension check: PASS (544)")

    # Check 2: No NaN values
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        logger.warning(f"  Found {nan_count} NaN values")
        checks_passed = False
    else:
        logger.info("  NaN check: PASS")

    # Check 3: L2 normalization
    norms = np.linalg.norm(X, axis=1)
    norm_close = np.allclose(norms, 1.0, atol=1e-5)
    if not norm_close:
        logger.warning(f"  Embeddings not L2 normalized (norm range: {norms.min():.4f} - {norms.max():.4f})")
        checks_passed = False
    else:
        logger.info(f"  L2 norm check: PASS (mean={norms.mean():.6f})")

    # Check 4: Reasonable variance
    variance = X.var(axis=0).mean()
    logger.info(f"  Mean variance: {variance:.6f}")
    if variance < 1e-6:
        logger.warning("  Very low variance - embeddings may be degenerate")
        checks_passed = False

    # Check 5: Component embeddings present
    required_cols = ["embedding", "proc_emb", "med_emb", "struct_emb"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.warning(f"  Missing columns: {missing}")
        checks_passed = False
    else:
        logger.info("  Column check: PASS")

    # Check 6: Procedure text present
    if "procedure_text" in df.columns:
        empty_procs = (df["procedure_text"] == "").sum()
        logger.info(f"  Empty procedure texts: {empty_procs}/{n_cases}")

    # Summary
    if checks_passed:
        logger.info("All verification checks PASSED")
    else:
        logger.warning("Some verification checks FAILED")

    return checks_passed


def main():
    """Main entry point."""
    args = parse_args()

    # Print banner
    print("=" * 60)
    print("PREOPERATIVE CASE EMBEDDING GENERATOR")
    print("=" * 60)
    print()

    # Log configuration
    logger.info("Configuration:")
    logger.info(f"  Case info DB: {args.caseinfo_db}")
    logger.info(f"  Medications DB: {args.meds_db}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Dimensions: proc={args.proc_dim}, med={args.med_dim}, struct={args.struct_dim}")
    logger.info(f"  Total dimension: {args.proc_dim + args.med_dim + args.struct_dim}")
    logger.info(f"  PCA fit batches: {args.pca_fit_batches} ({args.pca_fit_batches * args.batch_size:,} cases)")

    # Check database paths exist
    if not os.path.exists(args.caseinfo_db):
        logger.error(f"Case info database not found: {args.caseinfo_db}")
        sys.exit(1)
    if not os.path.exists(args.meds_db):
        logger.error(f"Medications database not found: {args.meds_db}")
        sys.exit(1)

    # Create configuration
    config = EmbeddingConfig(
        model_name=args.model,
        procedure_dim=args.proc_dim,
        medication_dim=args.med_dim,
        structured_dim=args.struct_dim,
        batch_size=args.batch_size,
        embed_batch_size=args.embed_batch_size,
        use_fp16=not args.no_fp16,
    )

    # Create embedder
    embedder = PreopCaseEmbedder(
        caseinfo_db=args.caseinfo_db,
        meds_db=args.meds_db,
        config=config,
    )

    # Determine case IDs to process
    case_ids = None
    debug_frac = None

    if args.case_ids:
        case_ids = load_case_ids_from_file(args.case_ids)
        logger.info(f"Loaded {len(case_ids)} case IDs from {args.case_ids}")
    elif args.debug:
        debug_frac = args.debug_frac
        logger.info(f"Debug mode: processing {debug_frac*100:.1f}% of cases")
    elif args.limit:
        case_ids = embedder.get_case_ids(limit=args.limit)
        logger.info(f"Limited to {len(case_ids)} cases")

    # Process
    t_start = time.time()
    df = embedder.process_all(
        output_path=args.output,
        batch_size=args.batch_size,
        debug_frac=debug_frac,
        case_ids=case_ids,
        pca_fit_batches=args.pca_fit_batches,
    )
    total_time = time.time() - t_start

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Cases processed: {len(df):,}")
    print(f"  Embedding dimension: {config.total_dim}")
    print(f"  Output file: {args.output}")
    print(f"  File size: {os.path.getsize(args.output) / 1e9:.2f} GB")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Rate: {len(df)/total_time:.0f} cases/second")
    print("=" * 60)

    # Verify if requested
    if args.verify:
        print()
        verify_embeddings(args.output)

    print()
    logger.info("Done!")


if __name__ == "__main__":
    main()
