#!/usr/bin/env python3
"""
Preoperative Case Embedding System
==================================

Creates comprehensive preoperative case embeddings using:
1. Semantic text embeddings for procedures (384 dims)
2. Semantic text embeddings for medications (64 dims)
3. Structured clinical data embeddings (96 dims)

Total: 544-dimensional case embedding, L2 normalized.

All data is from BEFORE surgery - perfect for case similarity,
UMAP visualization, and clustering.

Usage:
    from preop_case_embeddings import PreopCaseEmbedder

    embedder = PreopCaseEmbedder(caseinfo_db, meds_db)
    embedder.process_all("preop_embeddings.parquet")
"""

from __future__ import annotations

import gc
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import duckdb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    # Embedding dimensions
    procedure_dim: int = 384  # BGE-small output
    medication_dim: int = 64  # PCA reduced
    structured_dim: int = 96  # SVD reduced

    # Model
    model_name: str = "BAAI/bge-small-en-v1.5"

    # Processing
    batch_size: int = 10000  # Cases per batch
    embed_batch_size: int = 256  # Texts per embedding batch
    use_fp16: bool = True

    # Column names
    id_col: str = "MPOGCaseID"
    procedure_col: str = "ProcedureText"

    # Numerical features to use
    numerical_cols: Tuple[str, ...] = (
        "Age_Years",
        "BodyMassIndex",
        "Hemoglobin_Preop",
        "CreatMostRecent_Preop",
        "Glucose_Preop",
        "Platelets_Preop",
    )

    # Categorical features to use
    categorical_cols: Tuple[str, ...] = (
        "Sex",
        "ASAStatusClassification",
        "AdmissionType",
        "SurgicalService",
        "BodyRegion",
        "EmergStatusClass",
        "SmokTobaccoClass",
    )

    # Binary comorbidity features
    comorbidity_cols: Tuple[str, ...] = (
        "ComorbElixDiabetesComp",
        "ComorbElixCongHeartFailure",
        "ComorbElixRenalFailure",
        "ComorbElixChronicPulmDisease",
        "ComorbElixHypertensionComp",
    )

    @property
    def total_dim(self) -> int:
        """Total embedding dimension."""
        return self.procedure_dim + self.medication_dim + self.structured_dim


# =============================================================================
# DATABASE UTILITIES
# =============================================================================

def connect_db(path: str, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Connect to DuckDB database."""
    return duckdb.connect(path, read_only=read_only)


def get_tables(con: duckdb.DuckDBPyConnection) -> List[str]:
    """Get list of tables in database."""
    return con.execute("SHOW TABLES").fetchdf()["name"].tolist()


def get_columns(con: duckdb.DuckDBPyConnection, table: str) -> List[str]:
    """Get column names for a table."""
    return con.execute(f"DESCRIBE {table}").fetchdf()["column_name"].tolist()


def find_table_with_column(con: duckdb.DuckDBPyConnection, column: str) -> Optional[str]:
    """Find first table containing a specific column."""
    for table in get_tables(con):
        if column in get_columns(con, table):
            return table
    return None


def clean_text(text: Union[str, float, None], max_chars: int = 512) -> str:
    """Clean and normalize text, handling None/NaN."""
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()[:max_chars]


# =============================================================================
# EMBEDDING MODEL
# =============================================================================

class TextEmbedder:
    """
    Text embedding using sentence-transformers.

    Supports GPU acceleration and FP16 for efficiency.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: Optional[str] = None,
        use_fp16: bool = True,
    ):
        import torch
        from sentence_transformers import SentenceTransformer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and self.device == "cuda"

        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)

        if self.use_fp16:
            self.model.half()
            logger.info("Using FP16 precision")

        self.dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dim}, Device: {self.device}")

    def encode(
        self,
        texts: List[str],
        batch_size: int = 256,
        show_progress: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Whether to log progress
            normalize: Whether to L2 normalize embeddings

        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        import torch

        n = len(texts)
        if n == 0:
            return np.zeros((0, self.dim), dtype=np.float32)

        all_embeddings = []
        t0 = time.time()

        for i in range(0, n, batch_size):
            batch = texts[i : i + batch_size]

            with torch.no_grad():
                emb = self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    normalize_embeddings=normalize,
                    show_progress_bar=False,
                )

            all_embeddings.append(emb.astype(np.float32))

            if show_progress and (i + batch_size) % (batch_size * 10) == 0:
                done = min(i + batch_size, n)
                rate = done / (time.time() - t0)
                logger.info(f"  Embedded {done:,}/{n:,} texts ({rate:.0f}/s)")

            # Clear GPU cache periodically
            if self.device == "cuda" and (i // batch_size + 1) % 100 == 0:
                torch.cuda.empty_cache()

        X = np.vstack(all_embeddings)
        elapsed = time.time() - t0
        logger.info(f"  Embedded {n:,} texts -> {X.shape} in {elapsed:.1f}s")

        del all_embeddings
        gc.collect()

        return X

    def unload(self):
        """Free model memory."""
        import torch

        del self.model
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Embedding model unloaded")


# =============================================================================
# MAIN EMBEDDER CLASS
# =============================================================================

class PreopCaseEmbedder:
    """
    Preoperative case embedding generator.

    Creates 544-dimensional embeddings combining:
    - Procedure text embeddings (384 dims)
    - Medication profile embeddings (64 dims)
    - Structured clinical data embeddings (96 dims)
    """

    def __init__(
        self,
        caseinfo_db: str,
        meds_db: str,
        model_name: str = "BAAI/bge-small-en-v1.5",
        config: Optional[EmbeddingConfig] = None,
    ):
        """
        Initialize the embedder.

        Args:
            caseinfo_db: Path to case info DuckDB database
            meds_db: Path to preop medications DuckDB database
            model_name: Sentence transformer model name
            config: Optional configuration object
        """
        self.caseinfo_db = caseinfo_db
        self.meds_db = meds_db
        self.config = config or EmbeddingConfig(model_name=model_name)

        # Will be initialized lazily
        self._text_embedder: Optional[TextEmbedder] = None
        self._medication_pca: Optional[PCA] = None
        self._structured_pipeline: Optional[ColumnTransformer] = None
        self._structured_svd: Optional[TruncatedSVD] = None

        # Cache table names
        self._caseinfo_table: Optional[str] = None
        self._meds_table: Optional[str] = None

        logger.info(f"PreopCaseEmbedder initialized")
        logger.info(f"  Case info DB: {caseinfo_db}")
        logger.info(f"  Medications DB: {meds_db}")
        logger.info(f"  Total embedding dim: {self.config.total_dim}")

    def _get_text_embedder(self) -> TextEmbedder:
        """Get or create text embedder."""
        if self._text_embedder is None:
            self._text_embedder = TextEmbedder(
                model_name=self.config.model_name,
                use_fp16=self.config.use_fp16,
            )
        return self._text_embedder

    def _get_caseinfo_table(self, con: duckdb.DuckDBPyConnection) -> str:
        """Find the case info table."""
        if self._caseinfo_table is None:
            self._caseinfo_table = find_table_with_column(con, self.config.id_col)
            if self._caseinfo_table is None:
                raise ValueError(f"No table found with column {self.config.id_col}")
            logger.info(f"Using case info table: {self._caseinfo_table}")
        return self._caseinfo_table

    def _get_meds_table(self, con: duckdb.DuckDBPyConnection) -> str:
        """Find the medications table."""
        if self._meds_table is None:
            # Look for table with "71210" in name (MPOG preop meds concept)
            for table in get_tables(con):
                if "71210" in table:
                    self._meds_table = table
                    break
            if self._meds_table is None:
                # Fallback: find table with medication text column
                for table in get_tables(con):
                    cols = get_columns(con, table)
                    if any("text" in c.lower() for c in cols):
                        self._meds_table = table
                        break
            if self._meds_table is None:
                raise ValueError("No medications table found")
            logger.info(f"Using medications table: {self._meds_table}")
        return self._meds_table

    def get_case_ids(self, limit: Optional[int] = None) -> List[str]:
        """
        Get all case IDs from the database.

        Args:
            limit: Optional limit on number of cases

        Returns:
            List of case ID strings
        """
        con = connect_db(self.caseinfo_db)
        try:
            table = self._get_caseinfo_table(con)
            limit_clause = f"LIMIT {limit}" if limit else ""
            df = con.execute(
                f'SELECT "{self.config.id_col}" FROM {table} '
                f'WHERE "{self.config.procedure_col}" IS NOT NULL {limit_clause}'
            ).fetchdf()
            return df[self.config.id_col].astype(str).tolist()
        finally:
            con.close()

    def embed_procedures(
        self,
        case_ids: List[str],
        return_texts: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
        """
        Embed procedure text for given cases.

        Args:
            case_ids: List of case IDs
            return_texts: If True, also return the procedure texts

        Returns:
            numpy array of shape (n_cases, 384), optionally with texts
        """
        logger.info(f"Embedding procedures for {len(case_ids):,} cases")

        con = connect_db(self.caseinfo_db)
        try:
            table = self._get_caseinfo_table(con)

            # Fetch procedure text
            df = con.execute(
                f'SELECT "{self.config.id_col}", "{self.config.procedure_col}" '
                f"FROM {table} "
                f'WHERE "{self.config.id_col}" IN (SELECT * FROM UNNEST(?))',
                [case_ids],
            ).fetchdf()
        finally:
            con.close()

        # Reorder to match input case_ids
        df[self.config.id_col] = df[self.config.id_col].astype(str)
        df = df.set_index(self.config.id_col).loc[case_ids].reset_index()

        # Clean text
        texts = [clean_text(t) for t in df[self.config.procedure_col]]

        # Handle empty texts
        texts = [t if t else "unknown procedure" for t in texts]

        # Embed
        embedder = self._get_text_embedder()
        X = embedder.encode(texts, batch_size=self.config.embed_batch_size)

        if return_texts:
            return X, texts
        return X

    def embed_medications(self, case_ids: List[str]) -> np.ndarray:
        """
        Embed medication profiles for given cases.

        Concatenates all medication text per case, embeds with BGE,
        then reduces to 64 dims via PCA.

        Args:
            case_ids: List of case IDs

        Returns:
            numpy array of shape (n_cases, 64)
        """
        logger.info(f"Embedding medications for {len(case_ids):,} cases")

        con = connect_db(self.meds_db)
        try:
            table = self._get_meds_table(con)
            cols = get_columns(con, table)

            # Find text column
            text_col = next((c for c in cols if "text" in c.lower()), None)
            if text_col is None:
                logger.warning("No medication text column found, using zeros")
                return np.zeros(
                    (len(case_ids), self.config.medication_dim), dtype=np.float32
                )

            # Aggregate medication text per case
            df = con.execute(
                f"""
                SELECT "{self.config.id_col}",
                       STRING_AGG("{text_col}", ', ') AS med_text
                FROM {table}
                WHERE "{self.config.id_col}" IN (SELECT * FROM UNNEST(?))
                GROUP BY "{self.config.id_col}"
                """,
                [case_ids],
            ).fetchdf()
        finally:
            con.close()

        # Create mapping of case_id -> med_text
        df[self.config.id_col] = df[self.config.id_col].astype(str)
        med_dict = dict(zip(df[self.config.id_col], df["med_text"]))

        # Get medication text for each case (in order)
        med_texts = []
        for cid in case_ids:
            text = med_dict.get(cid, "")
            text = clean_text(text, max_chars=1024)  # Allow longer for med lists
            if not text:
                text = "no medications"
            med_texts.append(text)

        # Embed medication texts
        embedder = self._get_text_embedder()
        X_full = embedder.encode(med_texts, batch_size=self.config.embed_batch_size)

        # Reduce to target dimension via PCA
        if self._medication_pca is None:
            logger.info(
                f"Fitting medication PCA: {X_full.shape[1]} -> {self.config.medication_dim}"
            )
            n_components = min(
                self.config.medication_dim, X_full.shape[0] - 1, X_full.shape[1]
            )
            self._medication_pca = PCA(n_components=n_components, random_state=42)
            X_reduced = self._medication_pca.fit_transform(X_full)
        else:
            X_reduced = self._medication_pca.transform(X_full)

        # Pad if needed
        if X_reduced.shape[1] < self.config.medication_dim:
            padding = np.zeros(
                (X_reduced.shape[0], self.config.medication_dim - X_reduced.shape[1]),
                dtype=np.float32,
            )
            X_reduced = np.hstack([X_reduced, padding])

        # L2 normalize
        X_reduced = normalize(X_reduced, norm="l2").astype(np.float32)

        return X_reduced

    def embed_structured(self, case_ids: List[str]) -> np.ndarray:
        """
        Embed structured clinical features for given cases.

        Processes numerical, categorical, and binary features,
        then reduces to 96 dims via TruncatedSVD.

        Args:
            case_ids: List of case IDs

        Returns:
            numpy array of shape (n_cases, 96)
        """
        logger.info(f"Embedding structured features for {len(case_ids):,} cases")

        con = connect_db(self.caseinfo_db)
        try:
            table = self._get_caseinfo_table(con)
            existing_cols = set(get_columns(con, table))

            # Collect columns to fetch
            all_cols = [self.config.id_col]
            num_cols = [c for c in self.config.numerical_cols if c in existing_cols]
            cat_cols = [c for c in self.config.categorical_cols if c in existing_cols]
            bin_cols = [c for c in self.config.comorbidity_cols if c in existing_cols]

            all_cols.extend(num_cols + cat_cols + bin_cols)

            logger.info(
                f"  Using {len(num_cols)} numerical, "
                f"{len(cat_cols)} categorical, "
                f"{len(bin_cols)} binary features"
            )

            # Fetch data
            col_sql = ", ".join([f'"{c}"' for c in all_cols])
            df = con.execute(
                f"SELECT {col_sql} FROM {table} "
                f'WHERE "{self.config.id_col}" IN (SELECT * FROM UNNEST(?))',
                [case_ids],
            ).fetchdf()
        finally:
            con.close()

        # Reorder to match input case_ids
        df[self.config.id_col] = df[self.config.id_col].astype(str)
        df = df.set_index(self.config.id_col).loc[case_ids].reset_index()

        # Fill NaN with placeholders (no imputation)
        # Numerical: fill with 0
        for col in num_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(float)

        # Categorical: fill with "Unknown" category
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown").astype(str)

        # Binary: fill with 0
        for col in bin_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(float)

        # Build preprocessing pipeline
        if self._structured_pipeline is None:
            transformers = []

            if num_cols:
                transformers.append(
                    (
                        "num",
                        StandardScaler(),
                        num_cols,
                    )
                )

            if cat_cols:
                transformers.append(
                    (
                        "cat",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        cat_cols,
                    )
                )

            if bin_cols:
                transformers.append(
                    (
                        "bin",
                        "passthrough",
                        bin_cols,
                    )
                )

            if not transformers:
                logger.warning("No structured features available")
                return np.zeros(
                    (len(case_ids), self.config.structured_dim), dtype=np.float32
                )

            self._structured_pipeline = ColumnTransformer(
                transformers, remainder="drop"
            )
            X_raw = self._structured_pipeline.fit_transform(df)
            logger.info(f"  Raw structured features: {X_raw.shape}")

            # Fit SVD
            n_components = min(
                self.config.structured_dim, X_raw.shape[1] - 1, X_raw.shape[0] - 1
            )
            if n_components > 1:
                self._structured_svd = TruncatedSVD(
                    n_components=n_components, random_state=42
                )
                X_reduced = self._structured_svd.fit_transform(X_raw)
            else:
                X_reduced = X_raw
        else:
            X_raw = self._structured_pipeline.transform(df)
            if self._structured_svd is not None:
                X_reduced = self._structured_svd.transform(X_raw)
            else:
                X_reduced = X_raw

        # Pad if needed
        if X_reduced.shape[1] < self.config.structured_dim:
            padding = np.zeros(
                (X_reduced.shape[0], self.config.structured_dim - X_reduced.shape[1]),
                dtype=np.float32,
            )
            X_reduced = np.hstack([X_reduced, padding])

        # L2 normalize
        X_reduced = normalize(X_reduced, norm="l2").astype(np.float32)

        return X_reduced

    def embed_cases(
        self,
        case_ids: List[str],
        return_components: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generate full case embeddings.

        Args:
            case_ids: List of case IDs
            return_components: If True, also return individual embedding components

        Returns:
            Full embedding array (n_cases, 544), optionally with components
        """
        logger.info(f"Generating full embeddings for {len(case_ids):,} cases")

        # Get all components
        X_proc = self.embed_procedures(case_ids)
        X_med = self.embed_medications(case_ids)
        X_struct = self.embed_structured(case_ids)

        # Concatenate
        X_full = np.concatenate([X_proc, X_med, X_struct], axis=1)

        # Final L2 normalization
        X_full = normalize(X_full, norm="l2").astype(np.float32)

        logger.info(f"Full embedding shape: {X_full.shape}")

        if return_components:
            return X_full, X_proc, X_med, X_struct
        return X_full

    def process_all(
        self,
        output_path: str,
        batch_size: Optional[int] = None,
        debug_frac: Optional[float] = None,
        case_ids: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Process all cases and save to parquet.

        Args:
            output_path: Path for output parquet file
            batch_size: Number of cases per batch (default: config value)
            debug_frac: If set, only process this fraction of cases
            case_ids: If set, only process these specific case IDs

        Returns:
            DataFrame with case IDs and embeddings
        """
        batch_size = batch_size or self.config.batch_size

        # Get case IDs to process
        if case_ids is None:
            all_case_ids = self.get_case_ids()
            if debug_frac is not None:
                n_sample = max(1, int(len(all_case_ids) * debug_frac))
                all_case_ids = all_case_ids[:n_sample]
                logger.info(f"Debug mode: processing {n_sample:,} cases ({debug_frac*100:.1f}%)")
        else:
            all_case_ids = case_ids

        n_cases = len(all_case_ids)
        logger.info(f"Processing {n_cases:,} cases in batches of {batch_size:,}")

        # Process in batches
        results = []
        t_start = time.time()

        for batch_idx, i in enumerate(range(0, n_cases, batch_size)):
            batch_ids = all_case_ids[i : i + batch_size]
            batch_num = batch_idx + 1
            total_batches = (n_cases + batch_size - 1) // batch_size

            logger.info(f"Batch {batch_num}/{total_batches}: cases {i:,} to {i+len(batch_ids):,}")

            # Get embeddings and procedure text
            X_proc, texts = self.embed_procedures(batch_ids, return_texts=True)
            X_med = self.embed_medications(batch_ids)
            X_struct = self.embed_structured(batch_ids)

            # Concatenate and normalize
            X_full = np.concatenate([X_proc, X_med, X_struct], axis=1)
            X_full = normalize(X_full, norm="l2").astype(np.float32)

            # Build batch DataFrame
            batch_df = pd.DataFrame(
                {
                    "mpog_case_id": batch_ids,
                    "procedure_text": texts,
                    "embedding": list(X_full),
                    "proc_emb": list(X_proc),
                    "med_emb": list(X_med),
                    "struct_emb": list(X_struct),
                }
            )

            results.append(batch_df)

            # Log progress
            elapsed = time.time() - t_start
            cases_done = i + len(batch_ids)
            rate = cases_done / elapsed
            eta = (n_cases - cases_done) / rate if rate > 0 else 0

            logger.info(
                f"  Progress: {cases_done:,}/{n_cases:,} cases "
                f"({cases_done/n_cases*100:.1f}%) - "
                f"{rate:.0f} cases/s - ETA: {eta/60:.1f}min"
            )

            # Cleanup
            del X_proc, X_med, X_struct, X_full
            gc.collect()

        # Combine all batches
        logger.info("Combining batches...")
        df = pd.concat(results, ignore_index=True)

        # Save to parquet
        logger.info(f"Saving to {output_path}...")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        df.to_parquet(output_path, index=False)

        total_time = time.time() - t_start
        logger.info(
            f"Complete! Processed {n_cases:,} cases in {total_time/60:.1f} minutes"
        )
        logger.info(f"Output: {output_path}")

        # Unload model to free memory
        if self._text_embedder is not None:
            self._text_embedder.unload()
            self._text_embedder = None

        return df

    def fit_reducers(self, sample_size: int = 50000) -> None:
        """
        Pre-fit PCA and SVD reducers on a sample of data.

        This is useful for consistent dimensionality reduction
        across multiple batches.

        Args:
            sample_size: Number of cases to use for fitting
        """
        logger.info(f"Fitting reducers on {sample_size:,} case sample")

        # Get sample case IDs
        case_ids = self.get_case_ids(limit=sample_size)

        # Fit by running embed methods (they fit on first call)
        _ = self.embed_medications(case_ids)
        _ = self.embed_structured(case_ids)

        logger.info("Reducers fitted")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_embeddings(path: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load embeddings from parquet file.

    Args:
        path: Path to parquet file

    Returns:
        Tuple of (DataFrame, embedding matrix)
    """
    df = pd.read_parquet(path)
    X = np.vstack(df["embedding"].values)
    return df, X


def find_similar_cases(
    df: pd.DataFrame,
    query_idx: int,
    n_neighbors: int = 10,
    metric: str = "cosine",
) -> pd.DataFrame:
    """
    Find similar cases to a query case.

    Args:
        df: DataFrame with embeddings
        query_idx: Index of query case
        n_neighbors: Number of similar cases to return
        metric: Distance metric (cosine, euclidean)

    Returns:
        DataFrame of similar cases with distances
    """
    from sklearn.neighbors import NearestNeighbors

    X = np.vstack(df["embedding"].values)

    nn = NearestNeighbors(metric=metric, algorithm="brute")
    nn.fit(X)

    distances, indices = nn.kneighbors(
        X[query_idx : query_idx + 1], n_neighbors=n_neighbors + 1
    )

    # Exclude the query itself
    result = df.iloc[indices[0][1:]].copy()
    result["distance"] = distances[0][1:]
    result["similarity"] = 1.0 - result["distance"]

    return result.reset_index(drop=True)


if __name__ == "__main__":
    # Quick test
    print("PreopCaseEmbedder module loaded successfully")
    print(f"Default embedding dimension: {EmbeddingConfig().total_dim}")
