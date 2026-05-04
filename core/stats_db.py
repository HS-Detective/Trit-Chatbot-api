import os
import pandas as pd
from sqlalchemy import create_engine, inspect
from langchain_community.utilities.sql_database import SQLDatabase
import logging
import re

logger = logging.getLogger(__name__)

base_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(base_dir, "data")
db_path = os.path.join(data_dir, "stats.db")

# Use a persistent SQLite database file
DB_URI = f"sqlite:///{db_path}"
engine = create_engine(DB_URI, echo=False)

# Global structures to hold mapping logic
HS_CODE_MAPPING = []
MTI_NAME_MAP = {} # {code: label}

def init_db():
    """Reads the simplified CSVs from data/ directory."""
    global HS_CODE_MAPPING, MTI_NAME_MAP
    
    # Force reload if db file is empty or missing tables
    if os.path.exists(db_path) and os.path.getsize(db_path) == 0:
        logger.info("Found 0-byte stats.db. Deleting to force reload.")
        os.remove(db_path)

    # 1. Load MTI_TRADE_STATS into SQLite
    trade_stats_file = os.path.join(data_dir, "(DIMA)국가별 품목별 수출입 금액(MTI 4단위, 2022년-2024년 월별).csv")
    table_name = "MTI_TRADE_STATS"
    
    try:
        insp = inspect(engine)
        existing_tables = insp.get_table_names()
        
        if table_name in existing_tables:
            logger.info(f"Table {table_name} already exists in stats.db. Skipping CSV load.")
        elif os.path.exists(trade_stats_file):
            logger.info(f"Starting to load {trade_stats_file} into {table_name}...")
            chunksize = 50000
            numeric_cols = ["EXP_AMT", "IMP_AMT"]
            
            first_chunk = True
            # Using utf-8 as per test result
            for chunk in pd.read_csv(trade_stats_file, dtype=str, chunksize=chunksize, encoding='utf-8'):
                for col in numeric_cols:
                    if col in chunk.columns:
                        chunk[col] = chunk[col].astype(str).str.replace(',', '')
                        chunk[col] = pd.to_numeric(chunk[col], errors='coerce').fillna(0)
                        
                if first_chunk:
                    chunk.to_sql(table_name, engine, index=False, if_exists="replace")
                    first_chunk = False
                else:
                    chunk.to_sql(table_name, engine, index=False, if_exists="append")
                logger.info(f"Loaded a chunk for {table_name}")
            logger.info(f"Successfully finished loading {table_name}.")
        else:
            logger.warning(f"File not found: {trade_stats_file}")
            
    except Exception as e:
        logger.error(f"Error loading {trade_stats_file} into SQLite: {e}")

    # 2. Parse CODE_MAPPING_LIST and build MTI_NAME_MAP
    mapping_file = os.path.join(data_dir, "(DIMA)품목 코드 리스트_합본.csv")
    try:
        if os.path.exists(mapping_file):
            # Using utf-8 as per test result
            df_map = pd.read_csv(mapping_file, header=None, encoding='utf-8', on_bad_lines='skip')
            
            # Reset global variables
            HS_CODE_MAPPING = []
            MTI_NAME_MAP = {}
            
            # Regex to extract code and name from lines like "MTI 코드 0111 (곡물류)..."
            # Pattern matches "코드 [4자리숫자] ([품목명])"
            pattern = re.compile(r"코드\s*(\d{4})\s*\(([^)]+)\)")
            
            for val in df_map[0].dropna():
                line = str(val).strip()
                HS_CODE_MAPPING.append(line)
                
                match = pattern.search(line)
                if match:
                    code = match.group(1)
                    name = match.group(2).strip()
                    MTI_NAME_MAP[code] = name
                    
            logger.info(f"Loaded {len(HS_CODE_MAPPING)} mapping rules and {len(MTI_NAME_MAP)} MTI master names.")
        else:
            logger.warning(f"Mapping file not found: {mapping_file}")
    except Exception as e:
        logger.error(f"Error loading {mapping_file}: {e}")

# Initialize on import
init_db()

# Create LangChain SQLDatabase instance
db = SQLDatabase(engine)
