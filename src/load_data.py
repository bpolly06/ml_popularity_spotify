import pandas as pd

def load_raw_data(filepath="data/spotify_data.csv"):

    # Loading raw data
    df = pd.read_csv(filepath)
    return df