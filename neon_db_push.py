import pandas as pd 
from sqlalchemy import *

# db url ( connection string )
from config import DATABASE_URL

def raw_data_pushing():
    
    try:
        df = pd.read_csv("ai_job_dataset.csv")

        engine = create_engine(DATABASE_URL)  

        df.to_sql("ai_job_raw_dataset",engine,if_exists="replace",index=False)  
        print("raw data is pushed !!")
    except Exception as e  :
        print("something went wrong !!\n")
        print(e)

    return True


def fetch_raw_data():

        try:
            engine = create_engine(DATABASE_URL)  

            df = pd.read_sql("SELECT * FROM ai_job_raw_dataset", con=engine)
            print(df.head())

        except Exception as e:
             print(e)
             print("somthing went wrong !!")
        return True


def push_df_to_neon(df,table_name):
      
    try:

        engine = create_engine(DATABASE_URL)  

        df.to_sql(table_name,engine,if_exists="replace",index=False)  
        print("raw data is pushed !!")
    except Exception as e  :
        print("something went wrong !!\n")
        print(e)

    return True
