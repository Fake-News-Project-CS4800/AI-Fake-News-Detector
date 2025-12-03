from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import yaml

with open("./configs/api_config.yaml", "r") as f:
    config = yaml.safe_load(f)

database_url = config["database"]["url"]

engine = create_engine(database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# This file sets up the database connection and session for SQLAlchemy ORM