#!/usr/bin/env python3
"""
🏥 Surgical Case Similarity Engine
===================================
Finds clinically similar surgical cases using semantic embeddings.

Usage:
    # Interactive mode (default) - waits for input at the end
    python case_similarity.py --query "431fc882-49ec-ec11-818a-000c29909f52"
    
    # Batch mode for SLURM - exits after completion
    python case_similarity.py --query "..." --batch
    
    # Custom settings
    python case_similarity.py --query "..." --topn 5000 --results 50 --model bge-base

SLURM Example:
    sbatch --job-name=case_sim --time=2:00:00 --mem=32G --cpus-per-task=4 \\
           --output=case_sim_%j.log --wrap="python case_similarity.py --query '...' --batch"
"""

from __future__ import annotations
import argparse
import gc
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd

# sklearn
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize

# ============================================================================
# RICH TERMINAL OUTPUT
# ============================================================================

class Terminal:
    """Rich terminal output with colors and formatting."""
    
    # ANSI colors
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    
    # Box drawing
    BOX_TL = "╔"
    BOX_TR = "╗"
    BOX_BL = "╚"
    BOX_BR = "╝"
    BOX_H = "═"
    BOX_V = "║"
    
    @staticmethod
    def supports_color() -> bool:
        """Check if terminal supports color."""
        if os.environ.get("NO_COLOR"):
            return False
        if os.environ.get("FORCE_COLOR"):
            return True
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    
    def __init__(self):
        self.use_color = self.supports_color()
        self.width = min(os.get_terminal_size().columns if sys.stdout.isatty() else 80, 100)
    
    def _c(self, color: str, text: str) -> str:
        """Apply color if supported."""
        if self.use_color:
            return f"{color}{text}{self.RESET}"
        return text
    
    def header(self, title: str):
        """Print a big header box."""
        inner_width = self.width - 4
        print()
        print(self._c(self.CYAN, self.BOX_TL + self.BOX_H * (self.width - 2) + self.BOX_TR))
        print(self._c(self.CYAN, self.BOX_V) + " " * inner_width + " " + self._c(self.CYAN, self.BOX_V))
        
        # Center title
        padding = (inner_width - len(title)) // 2
        title_line = " " * padding + self._c(self.BOLD + self.WHITE, title) + " " * (inner_width - padding - len(title))
        print(self._c(self.CYAN, self.BOX_V) + " " + title_line + self._c(self.CYAN, self.BOX_V))
        
        print(self._c(self.CYAN, self.BOX_V) + " " * inner_width + " " + self._c(self.CYAN, self.BOX_V))
        print(self._c(self.CYAN, self.BOX_BL + self.BOX_H * (self.width - 2) + self.BOX_BR))
        print()
    
    def section(self, title: str):
        """Print a section header."""
        print()
        print(self._c(self.BOLD + self.BLUE, f"{'─' * 3} {title} {'─' * (self.width - len(title) - 5)}"))
    
    def checkpoint(self, msg: str):
        """Print a checkpoint with timestamp."""
        ts = datetime.now().strftime("%H:%M:%S")
        print()
        print(self._c(self.YELLOW, f"⏱  [{ts}]") + f" {self._c(self.BOLD, msg)}")
    
    def info(self, msg: str):
        """Print info message."""
        print(f"   {self._c(self.DIM, 'ℹ')}  {msg}")
    
    def success(self, msg: str):
        """Print success message."""
        print(f"   {self._c(self.GREEN, '✓')}  {msg}")
    
    def warn(self, msg: str):
        """Print warning message."""
        print(f"   {self._c(self.YELLOW, '⚠')}  {msg}")
    
    def error(self, msg: str):
        """Print error message."""
        print(f"   {self._c(self.RED, '✗')}  {msg}")
    
    def progress(self, current: int, total: int, prefix: str = "", suffix: str = ""):
        """Print progress bar."""
        pct = current / total if total > 0 else 0
        bar_width = 30
        filled = int(bar_width * pct)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        line = f"\r   {prefix} [{self._c(self.CYAN, bar)}] {current:,}/{total:,} {suffix}"
        print(line, end="", flush=True)
        if current >= total:
            print()
    
    def table(self, headers: List[str], rows: List[List], max_col_width: int = 20):
        """Print a formatted table."""
        # Calculate column widths
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], min(len(str(cell)), max_col_width))
        
        # Header
        header_line = " │ ".join(self._c(self.BOLD, str(h).ljust(widths[i])[:widths[i]]) for i, h in enumerate(headers))
        print(f"   {header_line}")
        print(f"   {'─┼─'.join('─' * w for w in widths)}")
        
        # Rows
        for row in rows:
            cells = []
            for i, cell in enumerate(row):
                s = str(cell)
                if len(s) > widths[i]:
                    s = s[:widths[i]-1] + "…"
                cells.append(s.ljust(widths[i]))
            print(f"   {' │ '.join(cells)}")
    
    def kv(self, key: str, value, indent: int = 3):
        """Print key-value pair."""
        print(" " * indent + self._c(self.DIM, f"{key}:") + f" {value}")
    
    def divider(self):
        """Print a divider line."""
        print(self._c(self.DIM, "─" * self.width))


# Global terminal instance
term = Terminal()


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Configuration for case similarity search."""
    
    # Database paths
    duckdb_path: str = "/nfs/turbo/umms-sachinkh/PCRC 247 Aghaeepour/case_info/pcrc_caseinfo.duckdb"
    meds_duckdb_path: str = "/nfs/turbo/umms-sachinkh/PCRC 247 Aghaeepour/pcrc_0247_preop_meds.duckdb"
    
    # Query
    query_id: str = ""
    
    # Column names
    id_col: str = "MPOGCaseID"
    text_col: str = "ProcedureText"
    
    # Retrieval
    stage1_topn: int = 10_000
    stage1_token_limit: int = 8
    stage1_token_minlen: int = 4
    
    # Gate filtering
    gate_fields: Tuple[str, ...] = ("AdmissionType", "SurgicalService", "BodyRegion")
    gate_mode: str = "soft"  # "soft", "hard", "off"
    soft_gate_min_match: int = 2
    
    # Embedding model
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_batch_size: int = 128
    use_fp16: bool = True
    
    # Structured features
    structured_svd_dim: int = 64
    
    # Fusion weights
    weight_procedure: float = 0.50
    weight_struct: float = 0.25
    weight_meds: float = 0.25
    
    # Output
    final_topn: int = 100
    out_dir: str = "./case_sim_outputs"
    
    # Runtime
    batch_mode: bool = False  # If True, don't wait for input at end


# ============================================================================
# DRUG CLASSIFICATION (SQL-based)
# ============================================================================

DRUG_CLASSES = {
    "beta_blocker": ["metoprolol", "atenolol", "carvedilol", "propranolol", "labetalol", "bisoprolol"],
    "ace_arb": ["lisinopril", "enalapril", "losartan", "valsartan", "ramipril", "olmesartan"],
    "anticoag": ["warfarin", "apixaban", "rivaroxaban", "eliquis", "xarelto", "heparin", "enoxaparin", "dabigatran"],
    "antiplatelet": ["aspirin", "clopidogrel", "plavix", "ticagrelor", "prasugrel"],
    "statin": ["atorvastatin", "simvastatin", "rosuvastatin", "pravastatin", "lovastatin"],
    "diuretic": ["furosemide", "lasix", "hydrochlorothiazide", "spironolactone", "bumetanide", "torsemide"],
    "insulin": ["insulin", "lantus", "humalog", "novolog", "glargine", "lispro", "aspart"],
    "metformin": ["metformin", "glucophage"],
    "opioid": ["oxycodone", "hydrocodone", "morphine", "fentanyl", "tramadol", "hydromorphone", "codeine"],
    "benzo": ["lorazepam", "diazepam", "alprazolam", "clonazepam", "midazolam"],
    "ssri_snri": ["sertraline", "fluoxetine", "escitalopram", "citalopram", "venlafaxine", "duloxetine"],
    "ppi": ["omeprazole", "pantoprazole", "esomeprazole", "lansoprazole"],
    "steroid": ["prednisone", "methylprednisolone", "dexamethasone", "hydrocortisone"],
    "thyroid": ["levothyroxine", "synthroid", "liothyronine"],
    "gabapentinoid": ["gabapentin", "pregabalin", "lyrica"],
}

ALL_DRUG_CLASSES = sorted(DRUG_CLASSES.keys())


# ============================================================================
# DATABASE UTILITIES
# ============================================================================

def connect_db(path: str) -> duckdb.DuckDBPyConnection:
    """Connect to DuckDB database."""
    return duckdb.connect(path, read_only=True)


def get_tables(con) -> List[str]:
    """Get list of tables."""
    return con.execute("SHOW TABLES").fetchdf()["name"].tolist()


def find_table(con, pattern: str) -> Optional[str]:
    """Find table containing pattern."""
    return next((t for t in get_tables(con) if pattern in t), None)


def get_columns(con, table: str) -> List[str]:
    """Get column names for table."""
    return con.execute(f"DESCRIBE {table}").fetchdf()["column_name"].tolist()


def clean_text(x, max_chars: int = 256) -> str:
    """Clean procedure text."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()[:max_chars]


def tokenize_for_like(text: str, minlen: int = 4, limit: int = 8) -> List[str]:
    """Extract tokens for LIKE matching."""
    toks = re.findall(r"[a-zA-Z]{4,}", text.lower())
    return sorted(set(toks), key=lambda t: -len(t))[:limit]


# ============================================================================
# EMBEDDING
# ============================================================================

class ProcedureEmbedder:
    """
    Semantic embedder for surgical procedures.
    
    Uses sentence-transformers with memory-efficient batching.
    """
    
    def __init__(self, model_name: str, device: str = "cpu", use_fp16: bool = True):
        import torch
        from sentence_transformers import SentenceTransformer
        
        self.device = device
        self.use_fp16 = use_fp16 and device == "cuda"
        
        term.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        
        if self.use_fp16:
            self.model.half()
            term.info("Using FP16 precision")
        
        self.dim = self.model.get_sentence_embedding_dimension()
        term.info(f"Embedding dimension: {self.dim}")
    
    def encode(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        """Encode texts with batching and progress tracking."""
        import torch
        
        n = len(texts)
        term.info(f"Encoding {n:,} procedures...")
        
        all_embeddings = []
        t0 = time.time()
        
        for i in range(0, n, batch_size):
            batch = texts[i:i + batch_size]
            
            with torch.no_grad():
                emb = self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
            
            all_embeddings.append(emb.astype(np.float32))
            
            # Progress
            done = min(i + batch_size, n)
            term.progress(done, n, prefix="Embedding", suffix=f"({done/(time.time()-t0):.0f}/s)")
            
            # Clear GPU cache
            if self.device == "cuda" and (i // batch_size + 1) % 50 == 0:
                torch.cuda.empty_cache()
        
        X = np.vstack(all_embeddings)
        term.success(f"Embedded {n:,} texts → shape {X.shape} in {time.time()-t0:.1f}s")
        
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
        term.info("Model unloaded")


# ============================================================================
# SEARCH STAGES
# ============================================================================

def stage1_get_candidates(con, cfg: Config, table: str) -> Tuple[List[str], pd.Series]:
    """
    Stage 1: Find candidate cases using keyword matching.
    
    Returns list of candidate IDs and the query row.
    """
    term.checkpoint("Stage 1: Finding candidates")
    
    # Get query case
    cols = [cfg.id_col, cfg.text_col] + list(cfg.gate_fields)
    cols_sql = ", ".join([f'"{c}"' for c in cols])
    qdf = con.execute(
        f'SELECT {cols_sql} FROM {table} WHERE "{cfg.id_col}" = ? LIMIT 1',
        [cfg.query_id]
    ).fetchdf()
    
    if len(qdf) == 0:
        raise ValueError(f"Query case not found: {cfg.query_id}")
    
    qrow = qdf.iloc[0]
    qtext = clean_text(qrow[cfg.text_col])
    
    term.info(f"Query procedure: {qtext[:70]}...")
    
    # Build search tokens
    toks = tokenize_for_like(qtext, cfg.stage1_token_minlen, cfg.stage1_token_limit)
    term.info(f"Search tokens: {toks}")
    
    # Build WHERE clause
    where = [f'"{cfg.text_col}" IS NOT NULL']
    params = []
    
    # Gate clause
    if cfg.gate_mode != "off":
        valid = {}
        for f in cfg.gate_fields:
            v = qrow.get(f)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                valid[f] = v
        
        if valid:
            if cfg.gate_mode == "soft":
                terms = [f'("{f}" = ?)::INT' for f in valid]
                where.append(f"({' + '.join(terms)}) >= {cfg.soft_gate_min_match}")
                params.extend(valid.values())
            else:  # hard
                for f, v in valid.items():
                    where.append(f'"{f}" = ?')
                    params.append(v)
    
    # LIKE clause for procedure text
    if toks:
        likes = [f'"{cfg.text_col}" ILIKE ?' for _ in toks]
        where.append(f"({' AND '.join(likes)})")
        params.extend([f"%{t}%" for t in toks])
    
    # Execute
    sql = f'SELECT "{cfg.id_col}" FROM {table} WHERE {" AND ".join(where)} LIMIT {cfg.stage1_topn}'
    
    t0 = time.time()
    df = con.execute(sql, params).fetchdf()
    
    ids = df[cfg.id_col].astype(str).tolist()
    
    # Ensure query is included
    if str(cfg.query_id) not in ids:
        ids = [str(cfg.query_id)] + ids
    
    term.success(f"Found {len(ids):,} candidates in {time.time()-t0:.1f}s")
    
    return ids, qrow


def stage2_fetch_data(con, cfg: Config, table: str, case_ids: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Stage 2: Fetch case data for candidates.
    
    Returns DataFrame and list of structured feature columns.
    """
    term.checkpoint("Stage 2: Fetching case data")
    
    # Columns to fetch
    wanted = [
        cfg.id_col, cfg.text_col,
        # Demographics
        "Age_Years", "Sex", "BodyMassIndex", "ASAStatusClassification",
        # Risk
        "EmergStatusClass", "SmokTobaccoClass",
        # Labs
        "Hemoglobin_Preop", "CreatMostRecent_Preop", "Glucose_Preop", "Platelets_Preop",
        # Comorbidities
        "ComorbElixDiabetesComp", "ComorbElixCongHeartFailure", "ComorbElixRenalFailure",
        "ComorbElixChronicPulmDisease", "ComorbElixHypertensionComp",
    ]
    
    # Filter to existing columns
    existing = set(get_columns(con, table))
    cols = [c for c in wanted if c in existing]
    
    term.info(f"Fetching {len(cols)} columns for {len(case_ids):,} cases")
    
    # Fetch in chunks
    col_sql = ", ".join([f'"{c}"' for c in cols])
    chunk_size = 5000
    parts = []
    
    t0 = time.time()
    for i in range(0, len(case_ids), chunk_size):
        chunk = case_ids[i:i + chunk_size]
        sql = f'SELECT {col_sql} FROM {table} WHERE "{cfg.id_col}" IN (SELECT * FROM UNNEST(?))'
        parts.append(con.execute(sql, [chunk]).fetchdf())
        term.progress(min(i + chunk_size, len(case_ids)), len(case_ids), prefix="Fetching")
    
    df = pd.concat(parts, ignore_index=True)
    df[cfg.id_col] = df[cfg.id_col].astype(str)
    df[cfg.text_col] = df[cfg.text_col].apply(lambda x: clean_text(x, 256))
    
    struct_cols = [c for c in cols if c not in [cfg.id_col, cfg.text_col]]
    
    term.success(f"Fetched {len(df):,} cases in {time.time()-t0:.1f}s")
    
    return df, struct_cols


def stage3_load_medications(cfg: Config, case_ids: List[str]) -> pd.DataFrame:
    """
    Stage 3: Load medication features via SQL aggregation.
    
    This does drug classification in SQL (memory efficient).
    """
    term.checkpoint("Stage 3: Loading medications")
    t0 = time.time()
    
    con = connect_db(cfg.meds_duckdb_path)
    
    try:
        # Find medications table
        table = find_table(con, "71210")
        if not table:
            term.warn("Medications table not found")
            return _empty_med_df(case_ids)
        
        cols = get_columns(con, table)
        text_col = next((c for c in cols if "Text" in c), None)
        if not text_col:
            term.warn("Medication text column not found")
            return _empty_med_df(case_ids)
        
        term.info(f"Table: {table}, Column: {text_col}")
        
        # Build SQL CASE statements for drug classes
        cases = []
        for cls, drugs in DRUG_CLASSES.items():
            conds = " OR ".join([f"lower({text_col}) LIKE '%{d}%'" for d in drugs])
            cases.append(f"MAX(CASE WHEN {conds} THEN 1 ELSE 0 END) AS has_{cls}")
        
        # Process in chunks
        chunk_size = 5000
        results = []
        
        for i in range(0, len(case_ids), chunk_size):
            chunk = case_ids[i:i + chunk_size]
            
            sql = f"""
                SELECT MPOGCaseID, COUNT(*) as med_count, {', '.join(cases)}
                FROM {table}
                WHERE MPOGCaseID IN (SELECT * FROM UNNEST(?))
                GROUP BY MPOGCaseID
            """
            
            results.append(con.execute(sql, [chunk]).fetchdf())
            term.progress(min(i + chunk_size, len(case_ids)), len(case_ids), prefix="Loading meds")
        
        # Combine
        if not results:
            return _empty_med_df(case_ids)
        
        df = pd.concat(results, ignore_index=True)
        df["n_drug_classes"] = df[[f"has_{c}" for c in ALL_DRUG_CLASSES]].sum(axis=1)
        
        # Merge to full list
        result = pd.DataFrame({"MPOGCaseID": case_ids}).merge(df, on="MPOGCaseID", how="left")
        
        # Fill NaN
        for col in [f"has_{c}" for c in ALL_DRUG_CLASSES] + ["med_count", "n_drug_classes"]:
            result[col] = result[col].fillna(0).astype(int)
        
        term.success(f"Loaded medications in {time.time()-t0:.1f}s")
        term.info(f"Cases with meds: {(result['med_count'] > 0).sum():,}")
        
        return result
        
    finally:
        con.close()


def _empty_med_df(case_ids: List[str]) -> pd.DataFrame:
    """Return empty medication DataFrame."""
    df = pd.DataFrame({"MPOGCaseID": case_ids})
    for c in ALL_DRUG_CLASSES:
        df[f"has_{c}"] = 0
    df["med_count"] = 0
    df["n_drug_classes"] = 0
    return df


def stage4_build_embeddings(df: pd.DataFrame, cfg: Config, struct_cols: List[str], device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stage 4: Build all embeddings.
    
    Returns (procedure_emb, structured_emb, medication_emb).
    """
    # 4a: Procedure embeddings (semantic)
    term.checkpoint("Stage 4a: Procedure embeddings")
    
    embedder = ProcedureEmbedder(cfg.embed_model, device, cfg.use_fp16)
    X_proc = embedder.encode(df[cfg.text_col].tolist(), cfg.embed_batch_size)
    embedder.unload()
    
    # 4b: Structured embeddings
    term.checkpoint("Stage 4b: Structured embeddings")
    
    if struct_cols:
        available = [c for c in struct_cols if c in df.columns]
        num_cols = [c for c in available if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in available if c not in num_cols]
        
        transformers = []
        if num_cols:
            transformers.append(("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler())
            ]), num_cols))
        if cat_cols:
            transformers.append(("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), cat_cols))
        
        if transformers:
            ct = ColumnTransformer(transformers, remainder="drop")
            X_raw = ct.fit_transform(df[available])
            
            n_comp = min(cfg.structured_svd_dim, X_raw.shape[1] - 1, X_raw.shape[0] - 1)
            if n_comp > 1:
                svd = TruncatedSVD(n_components=n_comp, random_state=42)
                X_struct = normalize(svd.fit_transform(X_raw), norm="l2").astype(np.float32)
            else:
                X_struct = normalize(X_raw, norm="l2").astype(np.float32)
        else:
            X_struct = np.zeros((len(df), 1), dtype=np.float32)
    else:
        X_struct = np.zeros((len(df), 1), dtype=np.float32)
    
    term.success(f"Structured embedding: {X_struct.shape}")
    
    # 4c: Medication embeddings
    term.checkpoint("Stage 4c: Medication embeddings")
    
    has_cols = [c for c in df.columns if c.startswith("has_")]
    X_med = df[has_cols].values.astype(np.float32)
    
    if "med_count" in df.columns:
        mc = df["med_count"].values.astype(np.float32)
        X_med = np.column_stack([X_med, (mc - mc.mean()) / (mc.std() + 1e-8)])
    if "n_drug_classes" in df.columns:
        nc = df["n_drug_classes"].values.astype(np.float32)
        X_med = np.column_stack([X_med, (nc - nc.mean()) / (nc.std() + 1e-8)])
    
    X_med = normalize(X_med, norm="l2").astype(np.float32)
    term.success(f"Medication embedding: {X_med.shape}")
    
    return X_proc, X_struct, X_med


def stage5_find_similar(
    X_proc: np.ndarray,
    X_struct: np.ndarray,
    X_med: np.ndarray,
    cfg: Config,
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Stage 5: Fuse embeddings and find similar cases.
    """
    term.checkpoint("Stage 5: Finding similar cases")
    
    # Find query index
    q_idx = df[df[cfg.id_col] == str(cfg.query_id)].index[0]
    term.info(f"Query index: {q_idx}")
    
    # Fuse embeddings
    Z = np.concatenate([
        cfg.weight_procedure * X_proc,
        cfg.weight_struct * X_struct,
        cfg.weight_meds * X_med
    ], axis=1)
    Z = normalize(Z, norm="l2")
    
    term.info(f"Fused embedding: {Z.shape}")
    
    # Find neighbors
    t0 = time.time()
    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(Z)
    dists, idxs = nn.kneighbors(Z[q_idx:q_idx+1], n_neighbors=min(cfg.final_topn + 1, len(df)))
    
    # Build results
    out = df.iloc[idxs[0]].copy().reset_index(drop=True)
    out["similarity"] = 1.0 - dists[0]
    out["proc_sim"] = cosine_similarity(X_proc[q_idx:q_idx+1], X_proc[idxs[0]])[0]
    out["struct_sim"] = cosine_similarity(X_struct[q_idx:q_idx+1], X_struct[idxs[0]])[0]
    out["med_sim"] = cosine_similarity(X_med[q_idx:q_idx+1], X_med[idxs[0]])[0]
    
    # Remove query case
    out = out[out[cfg.id_col] != str(cfg.query_id)].reset_index(drop=True).head(cfg.final_topn)
    
    term.success(f"Found {len(out)} similar cases in {time.time()-t0:.1f}s")
    
    return out


# ============================================================================
# MAIN SEARCH
# ============================================================================

def run_search(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Run the full similarity search pipeline.
    
    Returns (results_df, all_candidates_df, metadata).
    """
    import torch
    
    term.header("🏥 SURGICAL CASE SIMILARITY SEARCH")
    
    # Setup
    term.section("Configuration")
    term.kv("Query ID", cfg.query_id)
    term.kv("Embedding model", cfg.embed_model)
    term.kv("Candidates (max)", f"{cfg.stage1_topn:,}")
    term.kv("Results (max)", cfg.final_topn)
    term.kv("Weights", f"proc={cfg.weight_procedure}, struct={cfg.weight_struct}, meds={cfg.weight_meds}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    term.kv("Device", device)
    if device == "cuda":
        term.kv("GPU", torch.cuda.get_device_name(0))
    
    t_start = time.time()
    
    # Connect to main database
    con = connect_db(cfg.duckdb_path)
    
    try:
        # Find table
        tables = get_tables(con)
        table = next((t for t in tables if cfg.id_col in get_columns(con, t)), None)
        if not table:
            raise ValueError(f"Could not find table with column {cfg.id_col}")
        term.info(f"Using table: {table}")
        
        # Stage 1: Get candidates
        candidate_ids, query_row = stage1_get_candidates(con, cfg, table)
        
        # Stage 2: Fetch data
        df, struct_cols = stage2_fetch_data(con, cfg, table, candidate_ids)
        
    finally:
        con.close()
        gc.collect()
    
    # Stage 3: Medications
    df_meds = stage3_load_medications(cfg, df[cfg.id_col].tolist())
    df = df.merge(df_meds, on=cfg.id_col, how="left")
    del df_meds
    gc.collect()
    
    # Stage 4: Embeddings
    X_proc, X_struct, X_med = stage4_build_embeddings(df, cfg, struct_cols, device)
    
    # Stage 5: Search
    results = stage5_find_similar(X_proc, X_struct, X_med, cfg, df)
    
    # Cleanup
    del X_proc, X_struct, X_med
    gc.collect()
    
    # Metadata
    total_time = time.time() - t_start
    metadata = {
        "query_id": cfg.query_id,
        "query_procedure": str(query_row[cfg.text_col])[:200],
        "n_candidates": len(df),
        "n_results": len(results),
        "total_time_seconds": total_time,
        "embed_model": cfg.embed_model,
        "weights": {
            "procedure": cfg.weight_procedure,
            "structured": cfg.weight_struct,
            "medications": cfg.weight_meds,
        }
    }
    
    return results, df, metadata


def display_results(results: pd.DataFrame, df: pd.DataFrame, metadata: dict, cfg: Config):
    """Display search results in terminal."""
    
    term.header("📊 RESULTS")
    
    # Summary
    term.section("Summary")
    term.kv("Query", metadata["query_id"])
    term.kv("Query procedure", metadata["query_procedure"][:60] + "...")
    term.kv("Candidates searched", f"{metadata['n_candidates']:,}")
    term.kv("Similar cases found", metadata["n_results"])
    term.kv("Total time", f"{metadata['total_time_seconds']:.1f}s")
    
    # Top results table
    term.section("Top Similar Cases")
    
    headers = ["Rank", "Case ID", "Similarity", "Proc", "Struct", "Meds", "Age", "Meds#"]
    rows = []
    
    for i, row in results.head(15).iterrows():
        rows.append([
            i + 1,
            str(row[cfg.id_col])[:12] + "...",
            f"{row['similarity']:.3f}",
            f"{row['proc_sim']:.3f}",
            f"{row['struct_sim']:.3f}",
            f"{row['med_sim']:.3f}",
            int(row.get("Age_Years", 0)) if pd.notna(row.get("Age_Years")) else "-",
            int(row.get("med_count", 0)),
        ])
    
    term.table(headers, rows)
    
    # Detailed comparison with top match
    if len(results) > 0:
        term.section("Comparison: Query vs Top Match")
        
        query = df[df[cfg.id_col] == cfg.query_id].iloc[0]
        match = results.iloc[0]
        
        print()
        term.kv("Query procedure", str(query[cfg.text_col])[:80], indent=3)
        term.kv("Match procedure", str(match[cfg.text_col])[:80], indent=3)
        print()
        term.kv("Query age", query.get("Age_Years", "N/A"), indent=3)
        term.kv("Match age", match.get("Age_Years", "N/A"), indent=3)
        print()
        term.kv("Query meds", f"{query.get('med_count', 0)} medications, {query.get('n_drug_classes', 0)} drug classes", indent=3)
        term.kv("Match meds", f"{match.get('med_count', 0)} medications, {match.get('n_drug_classes', 0)} drug classes", indent=3)
        
        # Drug class comparison
        query_drugs = {c for c in ALL_DRUG_CLASSES if query.get(f"has_{c}", 0) == 1}
        match_drugs = {c for c in ALL_DRUG_CLASSES if match.get(f"has_{c}", 0) == 1}
        
        shared = query_drugs & match_drugs
        if shared:
            print()
            term.kv("Shared drug classes", ", ".join(sorted(shared)), indent=3)


def save_results(results: pd.DataFrame, metadata: dict, cfg: Config):
    """Save results to files."""
    
    term.section("Saving Results")
    
    os.makedirs(cfg.out_dir, exist_ok=True)
    
    # Save CSV
    csv_path = os.path.join(cfg.out_dir, f"similar_cases_{cfg.query_id[:8]}.csv")
    results.to_csv(csv_path, index=False)
    term.success(f"CSV: {csv_path}")
    
    # Save metadata
    import json
    meta_path = os.path.join(cfg.out_dir, f"metadata_{cfg.query_id[:8]}.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    term.success(f"Metadata: {meta_path}")


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> Config:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Find similar surgical cases using semantic embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--query", "-q",
        required=True,
        help="Query case ID (MPOGCaseID)"
    )
    
    parser.add_argument(
        "--db",
        default="/nfs/turbo/umms-sachinkh/PCRC 247 Aghaeepour/case_info/pcrc_caseinfo.duckdb",
        help="Path to main DuckDB database"
    )
    
    parser.add_argument(
        "--meds-db",
        default="/nfs/turbo/umms-sachinkh/PCRC 247 Aghaeepour/pcrc_0247_preop_meds.duckdb",
        help="Path to medications DuckDB database"
    )
    
    parser.add_argument(
        "--topn", "-n",
        type=int,
        default=10_000,
        help="Max candidates to consider (default: 10000)"
    )
    
    parser.add_argument(
        "--results", "-r",
        type=int,
        default=100,
        help="Number of results to return (default: 100)"
    )
    
    parser.add_argument(
        "--model", "-m",
        default="BAAI/bge-small-en-v1.5",
        choices=["BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"],
        help="Embedding model (default: bge-small)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Embedding batch size (reduce if OOM)"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="./case_sim_outputs",
        help="Output directory"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode - exit after completion (for SLURM)"
    )
    
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )
    
    args = parser.parse_args()
    
    if args.no_color:
        os.environ["NO_COLOR"] = "1"
    
    return Config(
        query_id=args.query,
        duckdb_path=args.db,
        meds_duckdb_path=args.meds_db,
        stage1_topn=args.topn,
        final_topn=args.results,
        embed_model=args.model,
        embed_batch_size=args.batch_size,
        out_dir=args.output,
        batch_mode=args.batch,
    )


def main():
    """Main entry point."""
    cfg = parse_args()
    
    try:
        # Run search
        results, df, metadata = run_search(cfg)
        
        # Display results
        display_results(results, df, metadata, cfg)
        
        # Save results
        save_results(results, metadata, cfg)
        
        # Final message
        term.header("✅ COMPLETE")
        
        if not cfg.batch_mode:
            # Interactive mode - wait for user
            term.section("Interactive Mode")
            print()
            print("   Results are loaded. You can explore them interactively.")
            print("   Variables available: results, df, metadata, cfg")
            print()
            print("   Press Ctrl+C or type 'exit' to quit.")
            print()
            
            # Start interactive loop
            while True:
                try:
                    cmd = input(term._c(term.CYAN, ">>> "))
                    cmd = cmd.strip()
                    
                    if cmd.lower() in ("exit", "quit", "q"):
                        break
                    elif cmd.lower() == "help":
                        print("\nCommands:")
                        print("  results.head(10)  - Show top 10 results")
                        print("  results.describe() - Summary statistics")
                        print("  results[results.med_sim > 0.5] - Filter results")
                        print("  metadata - Show search metadata")
                        print("  exit - Quit")
                        print()
                    elif cmd:
                        try:
                            result = eval(cmd)
                            if result is not None:
                                print(result)
                        except Exception as e:
                            print(f"Error: {e}")
                
                except KeyboardInterrupt:
                    print()
                    break
            
            print("\nGoodbye!")
        
    except Exception as e:
        term.error(f"Search failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
