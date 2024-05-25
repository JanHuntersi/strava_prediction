import os
import pandas as pd
from definitions import PATH_TO_RAW_WEATHER,PATH_TO_PREPROCESS_WEATHER


def preprocess_weather():
    print("Preprocessing Weather data")
    df = pd.read_csv(PATH_TO_RAW_WEATHER)
    print("Number of rows in raw data: ", len(df))


    df = df.dropna()

    # get rid of duplicates
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    print("Number of rows after removing duplicates and dropping na: ", len(df))

    # save to raw data and preprocess data
    df.to_csv(PATH_TO_RAW_WEATHER, index=False)


    df.to_csv(PATH_TO_PREPROCESS_WEATHER, index=False)


if __name__ == "__main__":
    preprocess_weather()
