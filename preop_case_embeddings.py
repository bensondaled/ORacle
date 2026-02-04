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
from sklearn.decomposition import IncrementalPCA
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
        self._medication_pca: Optional[IncrementalPCA] = None
        self._medication_pca_fitted: bool = False
        self._structured_pipeline: Optional[ColumnTransformer] = None
        self._structured_svd: Optional[IncrementalPCA] = None
        self._structured_svd_fitted: bool = False

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

            # Fetch procedure text (one per case, aggregated if multiple)
            df = con.execute(
                f'SELECT "{self.config.id_col}", '
                f'FIRST("{self.config.procedure_col}") AS "{self.config.procedure_col}" '
                f"FROM {table} "
                f'WHERE "{self.config.id_col}" IN (SELECT * FROM UNNEST(?)) '
                f'GROUP BY "{self.config.id_col}"',
                [case_ids],
            ).fetchdf()
        finally:
            con.close()

        # Reorder to match input case_ids (use dict for O(1) lookup)
        df[self.config.id_col] = df[self.config.id_col].astype(str)
        proc_dict = dict(zip(df[self.config.id_col], df[self.config.procedure_col]))

        # Build list in order of input case_ids
        proc_texts = [proc_dict.get(str(cid), "") for cid in case_ids]
        df = pd.DataFrame({
            self.config.id_col: case_ids,
            self.config.procedure_col: proc_texts,
        })

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

        Note: PCA must be fitted before calling this (via _embed_medications_raw
        and partial_fit during process_all).

        Args:
            case_ids: List of case IDs

        Returns:
            numpy array of shape (n_cases, 64)
        """
        # Get raw embeddings
        X_full = self._embed_medications_raw(case_ids)

        # Reduce via PCA (must be fitted)
        if self._medication_pca is None or not self._medication_pca_fitted:
            logger.warning("Medication PCA not fitted, returning zeros")
            return np.zeros((len(case_ids), self.config.medication_dim), dtype=np.float32)

        X_reduced = self._medication_pca.transform(X_full)

        # Pad if needed
        X_reduced = self._pad_to_dim(X_reduced, self.config.medication_dim)

        # L2 normalize
        X_reduced = normalize(X_reduced, norm="l2").astype(np.float32)

        return X_reduced

    def embed_structured(self, case_ids: List[str]) -> np.ndarray:
        """
        Embed structured clinical features for given cases.

        Processes numerical, categorical, and binary features,
        then reduces to 96 dims via IncrementalPCA.

        Note: SVD must be fitted before calling this (via _get_structured_raw
        and partial_fit during process_all).

        Args:
            case_ids: List of case IDs

        Returns:
            numpy array of shape (n_cases, 96)
        """
        # Get raw features
        X_raw = self._get_structured_raw(case_ids)

        # Reduce via SVD if available
        if self._structured_svd is None:
            X_reduced = X_raw
        elif not self._structured_svd_fitted:
            logger.warning("Structured SVD not fitted, returning padded raw")
            X_reduced = X_raw
        else:
            X_reduced = self._structured_svd.transform(X_raw)

        # Pad if needed
        X_reduced = self._pad_to_dim(X_reduced, self.config.structured_dim)

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
        pca_fit_batches: int = 10,
    ) -> pd.DataFrame:
        """
        Process all cases and save to parquet (single-pass, memory efficient).

        Uses incremental PCA fitting on first N batches, then streams remaining
        batches directly to parquet without re-embedding.

        Args:
            output_path: Path for output parquet file
            batch_size: Number of cases per batch (default: config value)
            debug_frac: If set, only process this fraction of cases
            case_ids: If set, only process these specific case IDs
            pca_fit_batches: Number of batches to fit PCA on (default: 10 = 100k cases)

        Returns:
            DataFrame with case IDs and embeddings
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

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
        total_batches = (n_cases + batch_size - 1) // batch_size
        logger.info(f"Processing {n_cases:,} cases in {total_batches} batches of {batch_size:,}")

        t_start = time.time()
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # =================================================================
        # SINGLE PASS: Fit PCA on first N batches, then transform all
        # =================================================================
        # We cache raw embeddings during PCA fitting phase to avoid re-embedding
        logger.info("=" * 60)
        logger.info(f"SINGLE PASS: Fit PCA on first {pca_fit_batches} batches, stream rest")
        logger.info("=" * 60)

        # Storage for PCA fitting phase
        fit_phase_med_embeddings = []  # Cache 384-dim embeddings during fit
        fit_phase_struct_raw = []  # Cache raw structured features during fit
        fit_phase_case_ids = []  # Track case IDs for fit phase

        pq_writer = None
        total_written = 0

        try:
            for batch_idx, i in enumerate(range(0, n_cases, batch_size)):
                batch_ids = all_case_ids[i : i + batch_size]
                batch_num = batch_idx + 1

                logger.info(f"Batch {batch_num}/{total_batches}: cases {i:,} to {i+len(batch_ids):,}")

                # Get procedure embeddings and text
                X_proc, texts = self.embed_procedures(batch_ids, return_texts=True)

                # PCA FITTING PHASE: Cache raw embeddings, partial_fit
                if batch_num <= pca_fit_batches:
                    logger.info(f"  [PCA Fitting] Batch {batch_num}/{pca_fit_batches}")

                    # Get raw medication embeddings (384-dim, no PCA yet)
                    X_med_raw = self._embed_medications_raw(batch_ids)
                    fit_phase_med_embeddings.append(X_med_raw)
                    fit_phase_case_ids.extend(batch_ids)

                    # Partial fit medication PCA
                    if self._medication_pca is None:
                        n_components = min(self.config.medication_dim, X_med_raw.shape[1])
                        self._medication_pca = IncrementalPCA(n_components=n_components)
                    self._medication_pca.partial_fit(X_med_raw)

                    # Get raw structured features and partial fit
                    X_struct_raw = self._get_structured_raw(batch_ids)
                    fit_phase_struct_raw.append(X_struct_raw)

                    if self._structured_svd is not None:
                        self._structured_svd.partial_fit(X_struct_raw)

                    # If this is the last fit batch, finalize PCA and process cached data
                    if batch_num == pca_fit_batches or batch_num == total_batches:
                        logger.info("  Finalizing PCA fitting...")
                        self._medication_pca_fitted = True
                        self._structured_svd_fitted = True

                        # Process all cached fit-phase data
                        logger.info(f"  Processing {len(fit_phase_case_ids):,} cached cases from fit phase...")

                        # Stack cached embeddings
                        all_med_raw = np.vstack(fit_phase_med_embeddings)
                        all_struct_raw = np.vstack(fit_phase_struct_raw)

                        # Transform with fitted PCA
                        X_med_all = self._medication_pca.transform(all_med_raw)
                        X_med_all = normalize(X_med_all, norm="l2").astype(np.float32)

                        if self._structured_svd is not None:
                            X_struct_all = self._structured_svd.transform(all_struct_raw)
                        else:
                            X_struct_all = all_struct_raw
                        X_struct_all = self._pad_to_dim(X_struct_all, self.config.structured_dim)
                        X_struct_all = normalize(X_struct_all, norm="l2").astype(np.float32)

                        # Re-embed procedures for fit phase cases (we already have them for current batch)
                        # Process in sub-batches to avoid re-loading all
                        offset = 0
                        for fit_batch_idx in range(batch_num):
                            fit_start = fit_batch_idx * batch_size
                            fit_end = min(fit_start + batch_size, len(fit_phase_case_ids))
                            fit_batch_ids = fit_phase_case_ids[fit_start:fit_end]
                            fit_batch_len = len(fit_batch_ids)

                            # Get corresponding slices
                            X_med_batch = X_med_all[offset:offset + fit_batch_len]
                            X_struct_batch = X_struct_all[offset:offset + fit_batch_len]

                            # Re-embed procedures for this sub-batch
                            X_proc_batch, texts_batch = self.embed_procedures(fit_batch_ids, return_texts=True)

                            # Concatenate and normalize
                            X_full = np.concatenate([X_proc_batch, X_med_batch, X_struct_batch], axis=1)
                            X_full = normalize(X_full, norm="l2").astype(np.float32)

                            # Write batch
                            pq_writer = self._write_batch_to_parquet(
                                pq_writer, output_path,
                                fit_batch_ids, texts_batch, X_full, X_proc_batch, X_med_batch, X_struct_batch
                            )
                            total_written += fit_batch_len
                            offset += fit_batch_len

                        # Clear caches
                        del fit_phase_med_embeddings, fit_phase_struct_raw, all_med_raw, all_struct_raw
                        del X_med_all, X_struct_all
                        fit_phase_med_embeddings = []
                        fit_phase_struct_raw = []
                        fit_phase_case_ids = []
                        gc.collect()

                # TRANSFORM PHASE: PCA is fitted, process directly
                else:
                    # Embed and transform medications
                    X_med = self.embed_medications(batch_ids)
                    X_struct = self.embed_structured(batch_ids)

                    # Concatenate and normalize
                    X_full = np.concatenate([X_proc, X_med, X_struct], axis=1)
                    X_full = normalize(X_full, norm="l2").astype(np.float32)

                    # Write batch
                    pq_writer = self._write_batch_to_parquet(
                        pq_writer, output_path,
                        batch_ids, texts, X_full, X_proc, X_med, X_struct
                    )
                    total_written += len(batch_ids)

                    del X_med, X_struct, X_full
                    gc.collect()

                # Log progress
                elapsed = time.time() - t_start
                rate = total_written / elapsed if elapsed > 0 else 0
                remaining = n_cases - total_written
                eta = remaining / rate if rate > 0 else 0

                logger.info(
                    f"  Written: {total_written:,}/{n_cases:,} cases "
                    f"({total_written/n_cases*100:.1f}%) - "
                    f"{rate:.0f} cases/s - ETA: {eta/60:.1f}min"
                )

        finally:
            if pq_writer is not None:
                pq_writer.close()

        total_time = time.time() - t_start
        logger.info("=" * 60)
        logger.info(
            f"Complete! Processed {n_cases:,} cases in {total_time/60:.1f} minutes"
        )
        logger.info(f"Output: {output_path}")
        logger.info(f"Rate: {n_cases/total_time:.0f} cases/second")
        logger.info("=" * 60)

        # Unload model to free memory
        if self._text_embedder is not None:
            self._text_embedder.unload()
            self._text_embedder = None

        # Return the saved dataframe
        return pd.read_parquet(output_path)

    def _embed_medications_raw(self, case_ids: List[str]) -> np.ndarray:
        """
        Embed medication profiles without PCA reduction.

        Returns raw 384-dim embeddings for PCA fitting.
        """
        logger.info(f"Embedding medications (raw) for {len(case_ids):,} cases")

        con = connect_db(self.meds_db)
        try:
            table = self._get_meds_table(con)
            cols = get_columns(con, table)

            # Find text column
            text_col = next((c for c in cols if "text" in c.lower()), None)
            if text_col is None:
                logger.warning("No medication text column found, using zeros")
                embedder = self._get_text_embedder()
                return np.zeros((len(case_ids), embedder.dim), dtype=np.float32)

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
            text = clean_text(text, max_chars=1024)
            if not text:
                text = "no medications"
            med_texts.append(text)

        # Embed medication texts (no PCA reduction)
        embedder = self._get_text_embedder()
        X_raw = embedder.encode(med_texts, batch_size=self.config.embed_batch_size)

        return X_raw

    def _get_structured_raw(self, case_ids: List[str]) -> np.ndarray:
        """
        Get raw structured features (after scaling/one-hot, before SVD).

        Used during PCA fitting phase.
        """
        logger.info(f"Getting structured features (raw) for {len(case_ids):,} cases")

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

            # Fetch data (DISTINCT to handle any duplicate rows)
            col_sql = ", ".join([f'"{c}"' for c in all_cols])
            df = con.execute(
                f"SELECT DISTINCT {col_sql} FROM {table} "
                f'WHERE "{self.config.id_col}" IN (SELECT * FROM UNNEST(?))',
                [case_ids],
            ).fetchdf()
        finally:
            con.close()

        # Ensure one row per case_id and reorder to match input
        df[self.config.id_col] = df[self.config.id_col].astype(str)
        df = df.drop_duplicates(subset=[self.config.id_col], keep='first')

        # Create lookup dict for each column
        df_dict = df.set_index(self.config.id_col).to_dict('index')

        # Rebuild DataFrame in order of input case_ids
        rows = []
        for cid in case_ids:
            cid_str = str(cid)
            if cid_str in df_dict:
                row = {self.config.id_col: cid_str, **df_dict[cid_str]}
            else:
                # Missing case - use defaults
                row = {self.config.id_col: cid_str}
                for col in all_cols[1:]:  # Skip id_col
                    row[col] = None
            rows.append(row)
        df = pd.DataFrame(rows)

        # Handle missing values
        missing_cols = []

        for col in num_cols:
            if col in df.columns:
                miss_col = f"{col}_missing"
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[miss_col] = df[col].isna().astype(float)
                df[col] = df[col].fillna(0).astype(float)
                missing_cols.append(miss_col)

        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown").astype(str)

        for col in bin_cols:
            if col in df.columns:
                miss_col = f"{col}_missing"
                df[miss_col] = df[col].isna().astype(float)
                df[col] = df[col].astype(str).str.lower()
                df[col] = df[col].replace({
                    "yes": "1", "true": "1",
                    "no": "0", "false": "0",
                    "nan": "0", "none": "0", "<na>": "0", "": "0"
                })
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)
                missing_cols.append(miss_col)

        # Build preprocessing pipeline if needed
        if self._structured_pipeline is None:
            transformers = []

            if num_cols:
                transformers.append(("num", StandardScaler(), num_cols))
            if cat_cols:
                transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
            if bin_cols:
                transformers.append(("bin", "passthrough", bin_cols))
            if missing_cols:
                transformers.append(("missing", "passthrough", missing_cols))

            if not transformers:
                return np.zeros((len(case_ids), self.config.structured_dim), dtype=np.float32)

            self._structured_pipeline = ColumnTransformer(transformers, remainder="drop")
            X_raw = self._structured_pipeline.fit_transform(df)

            # Initialize SVD
            n_components = min(self.config.structured_dim, X_raw.shape[1])
            if n_components > 1:
                self._structured_svd = IncrementalPCA(n_components=n_components)

            return X_raw.astype(np.float32)

        return self._structured_pipeline.transform(df).astype(np.float32)

    def _pad_to_dim(self, X: np.ndarray, target_dim: int) -> np.ndarray:
        """Pad array to target dimension if needed."""
        if X.shape[1] < target_dim:
            padding = np.zeros((X.shape[0], target_dim - X.shape[1]), dtype=np.float32)
            return np.hstack([X, padding])
        return X

    def _write_batch_to_parquet(
        self,
        writer,
        output_path: str,
        case_ids: List[str],
        texts: List[str],
        X_full: np.ndarray,
        X_proc: np.ndarray,
        X_med: np.ndarray,
        X_struct: np.ndarray,
    ):
        """Write a batch to parquet file (append mode)."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        batch_df = pd.DataFrame({
            "mpog_case_id": case_ids,
            "procedure_text": texts,
            "embedding": list(X_full),
            "proc_emb": list(X_proc),
            "med_emb": list(X_med),
            "struct_emb": list(X_struct),
        })

        table = pa.Table.from_pandas(batch_df)

        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)

        writer.write_table(table)
        return writer

    def fit_reducers(self, sample_size: int = 50000, batch_size: int = 10000) -> None:
        """
        Pre-fit IncrementalPCA reducers on a sample of data.

        Note: process_all() automatically fits during processing.
        This method is only needed if you want to fit separately.

        Args:
            sample_size: Number of cases to use for fitting
            batch_size: Batch size for fitting
        """
        logger.info(f"Fitting reducers on {sample_size:,} cases in batches of {batch_size}")

        # Get sample case IDs
        case_ids = self.get_case_ids(limit=sample_size)

        # Process in batches for IncrementalPCA
        for i in range(0, len(case_ids), batch_size):
            batch_ids = case_ids[i:i + batch_size]
            logger.info(f"  Fitting batch {i//batch_size + 1}: cases {i} to {i + len(batch_ids)}")

            # Get raw embeddings and partial_fit
            X_med_raw = self._embed_medications_raw(batch_ids)
            if self._medication_pca is None:
                n_components = min(self.config.medication_dim, X_med_raw.shape[1])
                self._medication_pca = IncrementalPCA(n_components=n_components)
            self._medication_pca.partial_fit(X_med_raw)

            X_struct_raw = self._get_structured_raw(batch_ids)
            if self._structured_svd is not None:
                self._structured_svd.partial_fit(X_struct_raw)

            gc.collect()

        # Mark as fitted so transforms work
        self._medication_pca_fitted = True
        self._structured_svd_fitted = True

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
