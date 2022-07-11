import pandas as pd
import preprocessing
import config


def main():
    df = pd.read_csv(config.test)
    preprocessing.ticket_cleaned(df)
    preprocessing.fill_nulls(df)
    preprocessing.relatives(df)
    df = pd.get_dummies(df, columns=["Sex", "Embarked"])
    df = df.drop(["Cabin", "Name", "Ticket", "SibSp", "Parch"], axis=1)
    #preprocessing.Scaler(df)
    #df=df.drop(["Survived"], axis=1)
    df.to_csv("./Input/preprocessed_test.csv", index=False)
    print(df.head())
    return df



if __name__ == "__main__":
    main()