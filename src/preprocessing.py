import pandas as pd
import numpy as np
import config
from sklearn.preprocessing import StandardScaler
  
def main():
    df = pd.read_csv(config.train)
    ticket_cleaned(df)
    fill_nulls(df)
    relatives(df)
    #Everything not in a function comes after this. I don't know why they don't work in functions smh
    df = pd.get_dummies(df, columns=["Sex", "Embarked"])
    df = df.drop(["Cabin", "Name", "PassengerId", "Ticket", "SibSp", "Parch"], axis=1)
    df.to_csv("./Input/preprocessed_train.csv", index=False)
    print(df.isnull().sum())
    print(df.dtypes)
    print(df.head(600))
    return df


def ticket_cleaned(df):
    df["Ticket"] =  df["Ticket"].apply(lambda x:str(x).split(" ")[-1])#Takes the last value of the split digits in an array
    df["Ticket"] =  df["Ticket"].apply(lambda x:str(x).replace('LINE', '0'))#replacing LINE with 0 will be used to represent absence
    return df

def fill_nulls(df):
    #df["Age"] =  df["Age"].fillna(df["Age"].mode())
    #df["Age"] =  df["Age"].fillna(df["Age"].mean())
    df["Age"] =  df["Age"].fillna(df["Age"].median())
    return df

#def Scaler(df):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    df["Pclass"] =  scaled[:, 0]
    df["Age"] =  scaled[:, 1]
    df["SibSp"] = scaled[:, 2]
    df["Parch"] = scaled[:, 3]
    df["Fare"] = scaled[:, 4]
    return df

def relatives(df):
    df["Relatives"] = df["SibSp"] + df["Parch"]
    df.loc[df["Relatives"] > 0, "Alone"] = 1
    df.loc[df["Relatives"] == 0, "Alone"] = 0
    return df
if __name__ == "__main__":
    main()