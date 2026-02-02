#!/usr/bin/env python3
"""
Medication Episode Analysis
===========================

Efficiently extract medication "episodes" (bundles given within 5-min windows)
from a large DuckDB medication table.

Uses DuckDB's SQL engine for:
- Time window detection via LAG/window functions
- Episode ID assignment
- Bundle extraction and counting

All heavy computation stays in DuckDB - Python just orchestrates.

Usage:
    python medication_episodes.py \
        --meds-db /path/to/medications.duckdb \
        --output episodes.parquet \
        --window-minutes 5

    # With preop embeddings for cluster stratification
    python medication_episodes.py \
        --meds-db /path/to/medications.duckdb \
        --embeddings /path/to/preop_embeddings.parquet \
        --output episodes.parquet
"""

import argparse
import logging
from pathlib import Path
from typing import Optional
import time

import duckdb
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# EPISODE DETECTION (all in DuckDB SQL)
# =============================================================================

def create_episodes_table(
    con: duckdb.DuckDBPyConnection,
    source_table: str = "medications",
    window_minutes: int = 5,
) -> str:
    """
    Create a table with episode assignments using DuckDB window functions.

    Episodes are groups of medications given within `window_minutes` of each other.
    Uses gap-based session detection: new episode starts when gap > window_minutes.

    Returns the name of the created table.
    """
    logger.info(f"Creating episodes table (window={window_minutes} min)...")

    # Step 1: Parse timestamps and compute gaps
    # Step 2: Assign episode IDs based on gaps > threshold
    # Step 3: Create episode table with medication sets

    sql = f"""
    CREATE OR REPLACE TABLE med_episodes AS
    WITH
    -- Parse timestamps and order by case + time
    parsed AS (
        SELECT
            MPOGCaseID as case_id,
            Medication as med_name,
            Medication_Route as route,
            Medication_Dose as dose,
            Medication_UOM as uom,
            strptime(DoseStart_DT, '%Y-%m-%d %H:%M:%S') as dose_time
        FROM {source_table}
        WHERE DoseStart_DT IS NOT NULL
          AND Medication IS NOT NULL
          AND MPOGCaseID IS NOT NULL
    ),

    -- Compute time gap from previous medication in same case
    with_gaps AS (
        SELECT
            *,
            COALESCE(
                EXTRACT(EPOCH FROM (dose_time - LAG(dose_time) OVER (
                    PARTITION BY case_id ORDER BY dose_time
                ))) / 60.0,
                999999  -- First med in case gets large gap (starts new episode)
            ) as gap_minutes
        FROM parsed
    ),

    -- Assign episode IDs: increment when gap > threshold
    with_episodes AS (
        SELECT
            *,
            SUM(CASE WHEN gap_minutes > {window_minutes} THEN 1 ELSE 0 END) OVER (
                PARTITION BY case_id ORDER BY dose_time
            ) as episode_id
        FROM with_gaps
    )

    SELECT
        case_id,
        episode_id,
        MIN(dose_time) as episode_start,
        MAX(dose_time) as episode_end,
        COUNT(*) as n_meds,
        LIST(med_name ORDER BY dose_time) as medications,
        LIST(DISTINCT med_name) as unique_meds,
        LIST(route ORDER BY dose_time) as routes,
        LIST(dose ORDER BY dose_time) as doses
    FROM with_episodes
    GROUP BY case_id, episode_id
    ORDER BY case_id, episode_id
    """

    con.execute(sql)

    # Get stats
    stats = con.execute("""
        SELECT
            COUNT(*) as n_episodes,
            COUNT(DISTINCT case_id) as n_cases,
            AVG(n_meds) as avg_meds_per_episode,
            MAX(n_meds) as max_meds_per_episode
        FROM med_episodes
    """).fetchone()

    logger.info(f"  Episodes: {stats[0]:,}")
    logger.info(f"  Cases: {stats[1]:,}")
    logger.info(f"  Avg meds/episode: {stats[2]:.1f}")
    logger.info(f"  Max meds/episode: {stats[3]}")

    return "med_episodes"


def extract_bundle_counts(
    con: duckdb.DuckDBPyConnection,
    min_bundle_size: int = 2,
    min_support_count: int = 100,
) -> pd.DataFrame:
    """
    Extract frequent medication bundles from episodes.

    A bundle is the set of unique medications in an episode.
    Returns bundles that appear in at least `min_support_count` episodes.
    """
    logger.info(f"Extracting bundles (min_size={min_bundle_size}, min_support={min_support_count})...")

    sql = f"""
    WITH bundle_strings AS (
        SELECT
            case_id,
            episode_id,
            -- Sort medications alphabetically and join to create bundle signature
            array_to_string(list_sort(unique_meds), ' + ') as bundle,
            len(unique_meds) as bundle_size
        FROM med_episodes
        WHERE len(unique_meds) >= {min_bundle_size}
    )
    SELECT
        bundle,
        bundle_size,
        COUNT(*) as episode_count,
        COUNT(DISTINCT case_id) as case_count
    FROM bundle_strings
    GROUP BY bundle, bundle_size
    HAVING COUNT(*) >= {min_support_count}
    ORDER BY episode_count DESC
    """

    df = con.execute(sql).fetchdf()
    logger.info(f"  Found {len(df):,} frequent bundles")

    return df


def extract_pairwise_cooccurrence(
    con: duckdb.DuckDBPyConnection,
    min_support_count: int = 100,
) -> pd.DataFrame:
    """
    Extract pairwise medication co-occurrence within episodes.

    More efficient than full bundle extraction for large datasets.
    """
    logger.info(f"Extracting pairwise co-occurrence (min_support={min_support_count})...")

    sql = f"""
    WITH episode_meds AS (
        SELECT
            case_id,
            episode_id,
            UNNEST(unique_meds) as med
        FROM med_episodes
    ),
    pairs AS (
        SELECT
            a.case_id,
            a.episode_id,
            LEAST(a.med, b.med) as med1,
            GREATEST(a.med, b.med) as med2
        FROM episode_meds a
        JOIN episode_meds b
            ON a.case_id = b.case_id
            AND a.episode_id = b.episode_id
            AND a.med < b.med
    )
    SELECT
        med1,
        med2,
        COUNT(*) as cooccur_count,
        COUNT(DISTINCT case_id) as case_count
    FROM pairs
    GROUP BY med1, med2
    HAVING COUNT(*) >= {min_support_count}
    ORDER BY cooccur_count DESC
    """

    df = con.execute(sql).fetchdf()
    logger.info(f"  Found {len(df):,} frequent pairs")

    return df


def get_medication_frequencies(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Get overall medication frequencies for context."""

    sql = """
    SELECT
        med,
        COUNT(*) as total_administrations,
        COUNT(DISTINCT case_id) as cases_with_med
    FROM (
        SELECT case_id, UNNEST(medications) as med FROM med_episodes
    )
    GROUP BY med
    ORDER BY total_administrations DESC
    """

    return con.execute(sql).fetchdf()


# =============================================================================
# CLUSTER STRATIFICATION
# =============================================================================

def stratify_by_clusters(
    con: duckdb.DuckDBPyConnection,
    embeddings_path: str,
    n_clusters: int = 10,
    min_support_count: int = 50,
) -> pd.DataFrame:
    """
    Stratify bundle frequencies by preoperative patient clusters.

    1. Load embeddings
    2. Cluster cases using K-means
    3. Join with episodes
    4. Compute bundle frequencies per cluster
    """
    logger.info(f"Stratifying by {n_clusters} clusters...")

    # Load embeddings
    emb_df = pd.read_parquet(embeddings_path)
    X = np.vstack(emb_df['embedding'].values)
    case_ids = emb_df['mpog_case_id'].values

    logger.info(f"  Loaded {len(emb_df):,} embeddings")

    # Cluster
    from sklearn.cluster import MiniBatchKMeans

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=10000)
    clusters = kmeans.fit_predict(X)

    # Create cluster mapping table in DuckDB
    cluster_df = pd.DataFrame({
        'case_id': case_ids,
        'cluster_id': clusters
    })
    con.execute("CREATE OR REPLACE TABLE case_clusters AS SELECT * FROM cluster_df")

    logger.info(f"  Clustered into {n_clusters} groups")

    # Compute bundle frequencies per cluster
    sql = f"""
    WITH bundle_strings AS (
        SELECT
            e.case_id,
            e.episode_id,
            c.cluster_id,
            array_to_string(list_sort(e.unique_meds), ' + ') as bundle,
            len(e.unique_meds) as bundle_size
        FROM med_episodes e
        JOIN case_clusters c ON e.case_id = c.case_id
        WHERE len(e.unique_meds) >= 2
    ),
    cluster_totals AS (
        SELECT cluster_id, COUNT(*) as total_episodes
        FROM bundle_strings
        GROUP BY cluster_id
    )
    SELECT
        b.bundle,
        b.bundle_size,
        b.cluster_id,
        COUNT(*) as episode_count,
        ct.total_episodes,
        COUNT(*) * 100.0 / ct.total_episodes as pct_of_cluster
    FROM bundle_strings b
    JOIN cluster_totals ct ON b.cluster_id = ct.cluster_id
    GROUP BY b.bundle, b.bundle_size, b.cluster_id, ct.total_episodes
    HAVING COUNT(*) >= {min_support_count}
    ORDER BY bundle, cluster_id
    """

    df = con.execute(sql).fetchdf()
    logger.info(f"  Found {len(df):,} cluster-bundle combinations")

    return df


def identify_universal_vs_specific_bundles(
    cluster_bundles: pd.DataFrame,
    n_clusters: int,
    universality_threshold: float = 0.8,
) -> tuple:
    """
    Classify bundles as universal (appear in most clusters) vs cluster-specific.

    Args:
        cluster_bundles: Output from stratify_by_clusters
        n_clusters: Total number of clusters
        universality_threshold: Fraction of clusters bundle must appear in to be "universal"

    Returns:
        (universal_bundles, specific_bundles) DataFrames
    """
    # Count how many clusters each bundle appears in
    bundle_cluster_counts = cluster_bundles.groupby('bundle').agg({
        'cluster_id': 'nunique',
        'episode_count': 'sum',
        'bundle_size': 'first'
    }).reset_index()
    bundle_cluster_counts.columns = ['bundle', 'n_clusters_present', 'total_episodes', 'bundle_size']

    # Classify
    threshold_clusters = int(n_clusters * universality_threshold)

    universal = bundle_cluster_counts[
        bundle_cluster_counts['n_clusters_present'] >= threshold_clusters
    ].sort_values('total_episodes', ascending=False)

    specific = bundle_cluster_counts[
        bundle_cluster_counts['n_clusters_present'] < threshold_clusters
    ].sort_values('total_episodes', ascending=False)

    logger.info(f"Universal bundles (in {threshold_clusters}+ clusters): {len(universal)}")
    logger.info(f"Cluster-specific bundles: {len(specific)}")

    return universal, specific


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(
    meds_db: str,
    output_dir: str,
    window_minutes: int = 5,
    min_bundle_size: int = 2,
    min_support_count: int = 100,
    embeddings_path: Optional[str] = None,
    n_clusters: int = 10,
):
    """Run the full medication episode analysis pipeline."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("MEDICATION EPISODE ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Database: {meds_db}")
    logger.info(f"Window: {window_minutes} minutes")
    logger.info(f"Output: {output_dir}")

    t_start = time.time()

    # Connect to database
    con = duckdb.connect(meds_db, read_only=True)

    # Check table
    tables = con.execute("SHOW TABLES").fetchdf()
    logger.info(f"Tables: {tables['name'].tolist()}")

    # Get row count
    n_rows = con.execute("SELECT COUNT(*) FROM medications").fetchone()[0]
    logger.info(f"Total medication records: {n_rows:,}")

    # Create episodes (in-database)
    # Note: We need write access for temp tables, so reconnect
    con.close()

    # Create a temp database for intermediate tables
    temp_db = str(output_path / "temp_episodes.duckdb")
    con = duckdb.connect(temp_db)

    # Attach source database
    con.execute(f"ATTACH '{meds_db}' AS source (READ_ONLY)")

    # Create episodes table
    create_episodes_table(con, "source.medications", window_minutes)

    # Extract bundles
    bundles_df = extract_bundle_counts(con, min_bundle_size, min_support_count)
    bundles_df.to_parquet(output_path / "frequent_bundles.parquet", index=False)
    logger.info(f"Saved: frequent_bundles.parquet")

    # Extract pairwise co-occurrence
    pairs_df = extract_pairwise_cooccurrence(con, min_support_count)
    pairs_df.to_parquet(output_path / "pairwise_cooccurrence.parquet", index=False)
    logger.info(f"Saved: pairwise_cooccurrence.parquet")

    # Get medication frequencies
    med_freq = get_medication_frequencies(con)
    med_freq.to_parquet(output_path / "medication_frequencies.parquet", index=False)
    logger.info(f"Saved: medication_frequencies.parquet")

    # Export episodes table
    episodes_df = con.execute("""
        SELECT case_id, episode_id, episode_start, episode_end, n_meds,
               array_to_string(unique_meds, '|') as meds_str
        FROM med_episodes
    """).fetchdf()
    episodes_df.to_parquet(output_path / "episodes.parquet", index=False)
    logger.info(f"Saved: episodes.parquet ({len(episodes_df):,} episodes)")

    # Cluster stratification (if embeddings provided)
    if embeddings_path and Path(embeddings_path).exists():
        cluster_bundles = stratify_by_clusters(
            con, embeddings_path, n_clusters, min_support_count // 2
        )
        cluster_bundles.to_parquet(output_path / "cluster_bundles.parquet", index=False)

        universal, specific = identify_universal_vs_specific_bundles(
            cluster_bundles, n_clusters
        )
        universal.to_parquet(output_path / "universal_bundles.parquet", index=False)
        specific.to_parquet(output_path / "cluster_specific_bundles.parquet", index=False)

        logger.info(f"Saved: cluster_bundles.parquet, universal_bundles.parquet, cluster_specific_bundles.parquet")

    con.close()

    # Clean up temp database
    Path(temp_db).unlink(missing_ok=True)

    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info(f"Pipeline complete in {elapsed:.1f}s")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)

    # Print top bundles
    print("\n" + "=" * 60)
    print("TOP 20 MEDICATION BUNDLES")
    print("=" * 60)
    print(bundles_df.head(20).to_string(index=False))

    print("\n" + "=" * 60)
    print("TOP 20 MEDICATION PAIRS")
    print("=" * 60)
    print(pairs_df.head(20).to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description="Medication episode analysis")

    parser.add_argument(
        "--meds-db", type=str, required=True,
        help="Path to medications DuckDB database"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="./medication_analysis",
        help="Output directory"
    )
    parser.add_argument(
        "--window-minutes", type=int, default=5,
        help="Time window for episode detection (default: 5)"
    )
    parser.add_argument(
        "--min-bundle-size", type=int, default=2,
        help="Minimum medications per bundle (default: 2)"
    )
    parser.add_argument(
        "--min-support", type=int, default=100,
        help="Minimum episode count for a bundle to be frequent (default: 100)"
    )
    parser.add_argument(
        "--embeddings", type=str, default=None,
        help="Path to preop embeddings parquet for cluster stratification"
    )
    parser.add_argument(
        "--n-clusters", type=int, default=10,
        help="Number of clusters for stratification (default: 10)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_pipeline(
        meds_db=args.meds_db,
        output_dir=args.output,
        window_minutes=args.window_minutes,
        min_bundle_size=args.min_bundle_size,
        min_support_count=args.min_support,
        embeddings_path=args.embeddings,
        n_clusters=args.n_clusters,
    )
