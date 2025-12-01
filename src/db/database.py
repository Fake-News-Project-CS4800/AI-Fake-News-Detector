# src/db/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import yaml

# Load config
with open("./configs/api_config.yaml", "r") as f:
    config = yaml.safe_load(f)

database_url = config["database"]["url"]

# Create SQLAlchemy engine
engine = create_engine(database_url)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()
