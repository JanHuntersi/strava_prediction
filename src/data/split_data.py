import pandas as pd
import os 
from definitions import PATH_TO_PROCESSED_IS_ACTIVE, PATH_TO_KUDOS_DATASET, PATH_TO_TEST_TRAIN

def split_data():

    is_active = pd.read_csv(PATH_TO_PROCESSED_IS_ACTIVE)
    kudos = pd.read_csv(PATH_TO_KUDOS_DATASET)

    if is_active.empty or kudos.empty:
        print("No data to split, exiting")
        return

    # split the data
    is_active_train = is_active.sample(frac=0.8, random_state=42)
    is_active_test = is_active.drop(is_active_train.index)

    kudos_train = kudos.sample(frac=0.9, random_state=42)
    kudos_test = kudos.drop(kudos_train.index)

    # save the data
    is_active_train.to_csv(os.path.join(PATH_TO_TEST_TRAIN, "is_active_train.csv"), index=False)
    is_active_test.to_csv(os.path.join(PATH_TO_TEST_TRAIN, "is_active_test.csv"), index=False)

    kudos_train.to_csv(os.path.join(PATH_TO_TEST_TRAIN, "kudos_train.csv"), index=False)
    kudos_test.to_csv(os.path.join(PATH_TO_TEST_TRAIN, "kudos_test.csv"), index=False)

    print("Data split completed. Saved to test_train folder")


if __name__ == "__main__":
    split_data()
