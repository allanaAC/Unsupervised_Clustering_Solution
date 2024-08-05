#this function is to load data 
import pandas as pd
import logging

#data_path = "/data/real_estate.csv"
def load_and_preprocess_data(data_path):
    try: 
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        logging.error(" Error in processing data: {}". format(e))
    