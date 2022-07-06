from cv2 import dft
import pandas as pd
import preprocessing
import config


def main():
    df = pd.read_csv(config.test)
    df = df.drop(["Cabin", "Name", "PassengerId"], axis=1)
    preprocessing.ticket_cleaned(df)
    print(df.head())
    return df



if __name__ == "__main__":
    main()