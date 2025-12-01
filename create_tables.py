# create_tables.py
from src.db.database import engine, Base
from src.db import models  # noqa: F401 (this just registers the model)

# Create all tables defined on Base subclasses
Base.metadata.create_all(bind=engine)
print("Tables created.")
