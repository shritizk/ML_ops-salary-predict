import pandas as pd 
from sqlalchemy import *

# db url ( connection string )
from config import DATABASE_URL

def raw_data_pushing():
    
    df = pd.read_csv("ai_job_dataset.csv")

    engine = create_engine(DATABASE_URL)  

    df.to_sql("ai_job_raw_dataset",engine,if_exists="replace",index=False)  
    return True

raw_data_pushing()
