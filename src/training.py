from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import config
import pandas as pd
from sklearn.preprocessing import StandardScaler

def main():
    train = pd.read_csv(config.preprocessed_train)
    test = pd.read_csv(config.preprocessed_test)
    submission = pd.read_csv(config.submission)
    training(train, test, submission)
    return train, test, submission

def training(train, test, submission):
    #Assigning independent and dependent variables
    X_train = train.drop(["Survived"], axis=1)
    y_train = train["Survived"]
    #Assigning independent variable 
    X_test = test.drop(["PassengerId"], axis=1)
    #Scaling X test and X train
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test) 


    #Training and predicting
    model = LGBMClassifier()
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    #Making a column for the predicted data
    test["Survived"] = y_predicted
    print(test.head(600))
    #Converting the results into a csv
    submission["PassengerId"] = test["PassengerId"]
    submission["Survived"] = test["Survived"]
    submission.to_csv("./Input/submission.csv", index=False)
    return submission




if __name__ == "__main__":
    main()