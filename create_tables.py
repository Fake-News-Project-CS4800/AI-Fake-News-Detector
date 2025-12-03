import os, sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.db.database import engine, Base
from src.db import models

print("Creating tables...")
Base.metadata.create_all(bind=engine)
print("Tables created successfully.")
