from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sqlalchemy import create_engine, inspect
from dataclasses import dataclass
from dotenv import load_dotenv
import os


DATABASE_URL=os.environ.get('DATABASE_URL')
engine = create_engine(DATABASE_URL)

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")
