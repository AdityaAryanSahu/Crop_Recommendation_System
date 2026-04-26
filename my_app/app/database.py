
# app/database.py (REPLACED)

from sqlmodel import create_engine, SQLModel, Session
import os

# Render provides the Neon URL via this standard environment variable.
# We include a generic format for local testing/fallback.
DATABASE_URL = os.environ.get(
    "DATABASE_URL", 
    "postgresql+psycopg2://postgres:password@localhost/mydb" 
)

# 🎯 FIX: Use the URL provided by Render/Neon
# Note: Render often automatically converts the standard postgres:// URL to 
# the required postgresql+psycopg2:// format, but we define the engine creation here.
engine = create_engine(DATABASE_URL, echo=False)

def create_db_and_tables():
    """Initializes the database and creates all defined tables."""
    # This remains the same as it's SQLAlchemy/SQLModel standard
    SQLModel.metadata.create_all(engine)

def get_session():
    """Dependency to provide a database session to FastAPI routes."""
    # This dependency remains the same, using the standard Session object
    with Session(engine) as session:
        yield session

# Note: The manual folder creation logic (DB_FOLDER.mkdir) is removed,
# as Neon/PostgreSQL is a remote service and does not need local folders.
