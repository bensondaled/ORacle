#!/usr/bin/env python3
"""
Visualize Preoperative Case Embeddings
======================================

Generate UMAP visualizations and analysis of preoperative case embeddings.

Usage:
    # Basic UMAP visualization
    python visualize_case_embeddings.py embeddings/preop_embeddings.parquet

    # With clustering
    python visualize_case_embeddings.py embeddings/preop_embeddings.parquet --cluster

    # Interactive HTML output
    python visualize_case_embeddings.py embeddings/preop_embeddings.parquet --interactive

    # Sample for faster visualization
    python visualize_case_embeddings.py embeddings/preop_embeddings.parquet --sample 50000

    # Color by specific column
    python visualize_case_embeddings.py embeddings/preop_embeddings.parquet --color-by SurgicalService

Examples:
    # Quick test
    python visualize_case_embeddings.py preop_embeddings.parquet --sample 10000

    # Full analysis with all plots
    python visualize_case_embeddings.py preop_embeddings.parquet --cluster --stats --save-umap
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.neighbors import NearestNeighbors

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_embeddings(path: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load embeddings from parquet file."""
    logger.info(f"Loading embeddings from {path}")
    df = pd.read_parquet(path)
    X = np.vstack(df["embedding"].values)
    logger.info(f"Loaded {len(df):,} cases with {X.shape[1]}-dim embeddings")
    return df, X


def load_case_metadata(
    caseinfo_db: str,
    case_ids: List[str],
) -> pd.DataFrame:
    """Load additional case metadata for visualization."""
    import duckdb

    logger.info(f"Loading metadata from {caseinfo_db}")

    con = duckdb.connect(caseinfo_db, read_only=True)
    try:
        # Find table
        tables = con.execute("SHOW TABLES").fetchdf()["name"].tolist()
        table = next(
            (t for t in tables if "MPOGCaseID" in con.execute(f"DESCRIBE {t}").fetchdf()["column_name"].tolist()),
            None
        )
        if table is None:
            logger.warning("Could not find case info table")
            return pd.DataFrame({"mpog_case_id": case_ids})

        # Fetch metadata columns
        cols = [
            "MPOGCaseID", "Age_Years", "Sex", "BodyMassIndex",
            "ASAStatusClassification", "AdmissionType", "SurgicalService",
            "BodyRegion", "EmergStatusClass"
        ]
        existing = set(con.execute(f"DESCRIBE {table}").fetchdf()["column_name"].tolist())
        cols = [c for c in cols if c in existing]

        col_sql = ", ".join([f'"{c}"' for c in cols])
        df = con.execute(
            f"SELECT {col_sql} FROM {table} "
            f"WHERE \"MPOGCaseID\" IN (SELECT * FROM UNNEST(?))",
            [case_ids]
        ).fetchdf()

        df["MPOGCaseID"] = df["MPOGCaseID"].astype(str)
        return df.rename(columns={"MPOGCaseID": "mpog_case_id"})

    finally:
        con.close()


def compute_umap(
    X: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    """Compute UMAP embedding."""
    from umap import UMAP

    logger.info(f"Computing UMAP ({X.shape[0]:,} points, {n_neighbors} neighbors)")
    t0 = time.time()

    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
        verbose=True,
    )
    X_umap = reducer.fit_transform(X)

    logger.info(f"UMAP completed in {time.time() - t0:.1f}s")
    return X_umap


def cluster_embeddings(
    X: np.ndarray,
    method: str = "kmeans",
    n_clusters: int = 20,
) -> np.ndarray:
    """Cluster embeddings."""
    logger.info(f"Clustering with {method} (k={n_clusters})")
    t0 = time.time()

    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X)
    elif method == "hdbscan":
        model = HDBSCAN(min_cluster_size=100, min_samples=10)
        labels = model.fit_predict(X)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
    logger.info(f"Found {n_clusters_found} clusters in {time.time() - t0:.1f}s")

    return labels


def plot_umap_static(
    X_umap: np.ndarray,
    labels: Optional[np.ndarray] = None,
    color_by: Optional[pd.Series] = None,
    title: str = "Preoperative Case Embeddings",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
) -> plt.Figure:
    """Create static UMAP scatter plot."""
    fig, ax = plt.subplots(figsize=figsize)

    # Determine colors
    if color_by is not None:
        # Categorical coloring
        categories = color_by.unique()
        n_cats = len(categories)

        if n_cats <= 20:
            cmap = plt.cm.get_cmap("tab20", n_cats)
        else:
            cmap = plt.cm.get_cmap("viridis", n_cats)

        cat_to_int = {cat: i for i, cat in enumerate(categories)}
        colors = [cat_to_int[c] for c in color_by]

        scatter = ax.scatter(
            X_umap[:, 0], X_umap[:, 1],
            c=colors, cmap=cmap, alpha=0.5, s=1
        )

        # Legend for top categories
        if n_cats <= 15:
            handles = [
                plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=cmap(cat_to_int[cat]), markersize=8, label=str(cat)[:20])
                for cat in list(categories)[:15]
            ]
            ax.legend(handles=handles, loc='upper right', fontsize=8)

    elif labels is not None:
        scatter = ax.scatter(
            X_umap[:, 0], X_umap[:, 1],
            c=labels, cmap="tab20", alpha=0.5, s=1
        )
        plt.colorbar(scatter, ax=ax, label="Cluster")
    else:
        ax.scatter(
            X_umap[:, 0], X_umap[:, 1],
            alpha=0.3, s=1, c="steelblue"
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    ax.set_aspect("equal")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {output_path}")

    return fig


def plot_umap_interactive(
    df: pd.DataFrame,
    X_umap: np.ndarray,
    color_by: Optional[str] = None,
    output_path: str = "umap_interactive.html",
) -> None:
    """Create interactive Plotly UMAP visualization."""
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        logger.error("Plotly not installed. Install with: pip install plotly")
        return

    logger.info("Creating interactive visualization...")

    # Add UMAP coordinates to DataFrame
    plot_df = df.copy()
    plot_df["umap_1"] = X_umap[:, 0]
    plot_df["umap_2"] = X_umap[:, 1]

    # Truncate procedure text for hover
    if "procedure_text" in plot_df.columns:
        plot_df["procedure_short"] = plot_df["procedure_text"].str[:100] + "..."

    # Create hover template
    hover_cols = ["mpog_case_id"]
    if "procedure_short" in plot_df.columns:
        hover_cols.append("procedure_short")

    if color_by and color_by in plot_df.columns:
        fig = px.scatter(
            plot_df,
            x="umap_1",
            y="umap_2",
            color=color_by,
            hover_data=hover_cols,
            title=f"Preoperative Case Embeddings (colored by {color_by})",
            opacity=0.5,
        )
    else:
        fig = px.scatter(
            plot_df,
            x="umap_1",
            y="umap_2",
            hover_data=hover_cols,
            title="Preoperative Case Embeddings",
            opacity=0.5,
        )

    fig.update_traces(marker=dict(size=3))
    fig.update_layout(
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        width=1200,
        height=900,
    )

    fig.write_html(output_path)
    logger.info(f"Saved interactive plot to {output_path}")


def compute_embedding_stats(df: pd.DataFrame, X: np.ndarray) -> Dict:
    """Compute statistics about the embeddings."""
    stats = {}

    # Basic stats
    stats["n_cases"] = len(df)
    stats["embedding_dim"] = X.shape[1]

    # Norm statistics
    norms = np.linalg.norm(X, axis=1)
    stats["norm_mean"] = float(norms.mean())
    stats["norm_std"] = float(norms.std())

    # Variance per dimension
    variances = X.var(axis=0)
    stats["variance_mean"] = float(variances.mean())
    stats["variance_std"] = float(variances.std())

    # Component-wise stats (if available)
    if "proc_emb" in df.columns:
        X_proc = np.vstack(df["proc_emb"].values)
        stats["proc_emb_variance"] = float(X_proc.var(axis=0).mean())
    if "med_emb" in df.columns:
        X_med = np.vstack(df["med_emb"].values)
        stats["med_emb_variance"] = float(X_med.var(axis=0).mean())
    if "struct_emb" in df.columns:
        X_struct = np.vstack(df["struct_emb"].values)
        stats["struct_emb_variance"] = float(X_struct.var(axis=0).mean())

    # Pairwise similarity sample
    sample_idx = np.random.choice(len(X), min(1000, len(X)), replace=False)
    X_sample = X[sample_idx]
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(X_sample)
    np.fill_diagonal(sim_matrix, 0)
    stats["avg_pairwise_similarity"] = float(sim_matrix.mean())
    stats["max_pairwise_similarity"] = float(sim_matrix.max())

    return stats


def find_nearest_neighbors(
    df: pd.DataFrame,
    X: np.ndarray,
    query_idx: int,
    n_neighbors: int = 10,
) -> pd.DataFrame:
    """Find nearest neighbors for a query case."""
    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(X)

    distances, indices = nn.kneighbors(
        X[query_idx:query_idx+1],
        n_neighbors=n_neighbors + 1
    )

    result = df.iloc[indices[0][1:]].copy()
    result["distance"] = distances[0][1:]
    result["similarity"] = 1.0 - result["distance"]

    return result.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize preoperative case embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "input",
        type=str,
        help="Path to embeddings parquet file",
    )

    # Sampling
    parser.add_argument(
        "--sample",
        type=int,
        help="Sample N cases for visualization (faster)",
    )

    # UMAP parameters
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors parameter (default: 15)",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter (default: 0.1)",
    )

    # Clustering
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="Perform clustering on embeddings",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=20,
        help="Number of clusters for KMeans (default: 20)",
    )
    parser.add_argument(
        "--cluster-method",
        type=str,
        default="kmeans",
        choices=["kmeans", "hdbscan"],
        help="Clustering method (default: kmeans)",
    )

    # Coloring
    parser.add_argument(
        "--color-by",
        type=str,
        help="Column name to color points by",
    )
    parser.add_argument(
        "--caseinfo-db",
        type=str,
        help="Path to case info DB for additional metadata",
    )

    # Output
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="viz_output",
        help="Output directory for plots (default: viz_output)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Generate interactive Plotly HTML",
    )
    parser.add_argument(
        "--save-umap",
        action="store_true",
        help="Save UMAP coordinates to parquet",
    )

    # Analysis
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Compute and print embedding statistics",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Case ID to find neighbors for",
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=10,
        help="Number of nearest neighbors to show (default: 10)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load embeddings
    df, X = load_embeddings(args.input)

    # Sample if requested
    if args.sample and args.sample < len(df):
        logger.info(f"Sampling {args.sample:,} cases")
        idx = np.random.choice(len(df), args.sample, replace=False)
        df = df.iloc[idx].reset_index(drop=True)
        X = X[idx]

    # Load additional metadata if requested
    if args.caseinfo_db and args.color_by:
        meta_df = load_case_metadata(
            args.caseinfo_db,
            df["mpog_case_id"].tolist()
        )
        df = df.merge(meta_df, on="mpog_case_id", how="left")

    # Compute stats if requested
    if args.stats:
        print("\n" + "=" * 50)
        print("EMBEDDING STATISTICS")
        print("=" * 50)
        stats = compute_embedding_stats(df, X)
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")
        print("=" * 50 + "\n")

    # Query for neighbors if requested
    if args.query:
        query_mask = df["mpog_case_id"] == args.query
        if query_mask.sum() == 0:
            logger.error(f"Case ID not found: {args.query}")
        else:
            query_idx = query_mask.values.argmax()
            neighbors = find_nearest_neighbors(df, X, query_idx, args.n_results)

            print("\n" + "=" * 50)
            print(f"NEAREST NEIGHBORS FOR {args.query}")
            print("=" * 50)
            for i, row in neighbors.iterrows():
                proc = row.get("procedure_text", "")[:60] if "procedure_text" in row else ""
                print(f"  {i+1}. {row['mpog_case_id']} (sim={row['similarity']:.3f})")
                if proc:
                    print(f"     {proc}...")
            print("=" * 50 + "\n")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Compute UMAP
    logger.info("Computing UMAP projection...")
    X_umap = compute_umap(
        X,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
    )

    # Clustering
    labels = None
    if args.cluster:
        labels = cluster_embeddings(
            X,
            method=args.cluster_method,
            n_clusters=args.n_clusters,
        )
        df["cluster"] = labels

    # Determine color variable
    color_series = None
    if args.color_by and args.color_by in df.columns:
        color_series = df[args.color_by]
    elif args.cluster:
        color_series = df["cluster"]

    # Static plot
    output_path = os.path.join(args.output_dir, "umap_embeddings.png")
    plot_umap_static(
        X_umap,
        labels=labels,
        color_by=color_series,
        title=f"Preoperative Case Embeddings (n={len(df):,})",
        output_path=output_path,
    )

    # Interactive plot
    if args.interactive:
        html_path = os.path.join(args.output_dir, "umap_interactive.html")
        plot_umap_interactive(
            df,
            X_umap,
            color_by=args.color_by if args.color_by in df.columns else None,
            output_path=html_path,
        )

    # Save UMAP coordinates
    if args.save_umap:
        umap_df = df[["mpog_case_id"]].copy()
        umap_df["umap_1"] = X_umap[:, 0]
        umap_df["umap_2"] = X_umap[:, 1]
        if labels is not None:
            umap_df["cluster"] = labels

        umap_path = os.path.join(args.output_dir, "umap_coordinates.parquet")
        umap_df.to_parquet(umap_path, index=False)
        logger.info(f"Saved UMAP coordinates to {umap_path}")

    # Summary
    print("\n" + "=" * 50)
    print("VISUALIZATION COMPLETE")
    print("=" * 50)
    print(f"  Cases visualized: {len(df):,}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Static plot: {output_path}")
    if args.interactive:
        print(f"  Interactive plot: {html_path}")
    if args.save_umap:
        print(f"  UMAP coordinates: {umap_path}")
    if labels is not None:
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"  Clusters found: {n_clusters}")
    print("=" * 50)

    logger.info("Done!")


if __name__ == "__main__":
    main()
