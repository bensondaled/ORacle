#!/usr/bin/env python3
"""Quick case info stats."""

import duckdb

DB_PATH = "/nfs/turbo/umms-sachinkh/PCRC 247 Aghaeepour/case_info/pcrc_caseinfo.duckdb"

con = duckdb.connect(DB_PATH, read_only=True)

# Find table
tables = con.execute("SHOW TABLES").fetchdf()["name"].tolist()
print(f"Tables: {tables}")

# Use first table (or adjust if needed)
table = tables[0]
print(f"\nUsing table: {table}\n")

# Show columns
cols = con.execute(f"DESCRIBE {table}").fetchdf()["column_name"].tolist()
print(f"Columns: {cols[:20]}...")  # First 20

# Total cases
total = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
print(f"\n{'='*50}")
print(f"Total cases: {total:,}")

# Sex distribution
print(f"\n{'='*50}")
print("Sex distribution:")
sex_col = next((c for c in cols if 'sex' in c.lower()), None)
if sex_col:
    result = con.execute(f'SELECT "{sex_col}", COUNT(*) as n FROM {table} GROUP BY "{sex_col}" ORDER BY n DESC').fetchdf()
    print(result.to_string(index=False))
else:
    print("  No sex column found")

# Age > 18
print(f"\n{'='*50}")
print("Age distribution:")
age_col = next((c for c in cols if 'age' in c.lower()), None)
if age_col:
    adults = con.execute(f'SELECT COUNT(*) FROM {table} WHERE "{age_col}" >= 18').fetchone()[0]
    minors = con.execute(f'SELECT COUNT(*) FROM {table} WHERE "{age_col}" < 18').fetchone()[0]
    print(f"  Adults (>=18): {adults:,}")
    print(f"  Minors (<18):  {minors:,}")
else:
    print("  No age column found")

# Race distribution
print(f"\n{'='*50}")
print("Race distribution:")
race_col = next((c for c in cols if 'race' in c.lower()), None)
if race_col:
    result = con.execute(f'SELECT "{race_col}", COUNT(*) as n FROM {table} GROUP BY "{race_col}" ORDER BY n DESC').fetchdf()
    print(result.to_string(index=False))
else:
    print("  No race column found")

con.close()
print(f"\n{'='*50}")
