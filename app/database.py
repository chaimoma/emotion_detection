from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
import os

load_dotenv()
# Define  Database URL
USER_DB=os.getenv("USER_DB")
USER_DB_PASSWORD=os.getenv("USER_DB_PASSWORD")
DB_HOST=os.getenv("DB_HOST")
DB_NAME=os.getenv("DB_NAME")  
PORT=os.getenv("PORT")

SQLALCHEMY_DATABASE_URL=f"postgresql+psycopg2://{USER_DB}:{USER_DB_PASSWORD}@{DB_HOST}:{PORT}/{DB_NAME}"
engine=create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False,autoflush=False,bind=engine)
Base = declarative_base()

