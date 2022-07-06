import pandas as pd
import numpy as np
import config
from sklearn.preprocessing import LabelEncoder
  
def main():
    df = pd.read_csv(config.train)
    ticket_cleaned(df)
    #cols_encoded(df)
    fill_nulls(df)
    df = df.drop(["Cabin", "Name", "PassengerId", "Embarked", "Sex", "Ticket"], axis=1)
    df.to_csv("./Input/preprocessed_train.csv", index=False)
    print(df.isnull().sum())
    print(df.dtypes)
    print(df.head(600))
    return df


def ticket_cleaned(df):
    df["Ticket"] =  df["Ticket"].apply(lambda x:str(x).split(" ")[-1])#Takes the last value of the split digits in an array
    df["Ticket"] =  df["Ticket"].apply(lambda x:str(x).replace('LINE', '0'))#replacing LINE with 0 will be used to represent absence
    return df

#def cols_encoded(df):
    encoder = LabelEncoder()
    df["embarked_encoded"] = encoder.fit_transform(df["Embarked"])
    df["sex_encoded"] = encoder.fit_transform(df["Sex"])
    return df

def fill_nulls(df):
    #df["Age"] =  df["Age"].fillna(df["Age"].mode())
    #df["Age"] =  df["Age"].fillna(df["Age"].mean())
    df["Age"] =  df["Age"].fillna(df["Age"].median())
    return df


if __name__ == "__main__":
    main()