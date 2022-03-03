import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn import 
training = pd.read_csv('/home/mwanikii/Documents/Data Science/train.csv')
test = pd.read_csv('/home/mwanikii/Documents/Data Science/test.csv')

training['train_test'] = 1
test['train_test'] = 0
test['Survived'] = np.NaN
 
all_data = pd.concat([training, test])


all_data.columns


#Removing the females from the data
df = training
df = df.drop(df.index[df['Sex'] == 'female'], inplace = True)

print (training.head())

#Checking for null counts and datatypes
training.info()

#Getting more numerical data
training.describe()

#separating numeriacal columns
training.describe().columns

#Looking at the numerical and categorical values separately
df_num = training[['Age', 'Parch', 'SibSp', 'Fare']]
df_cat = training[['Survived', 'Pclass', 'Sex', 'Ticket', 'Embarked', 'Cabin']]

#Distribution for numerical values
for i in df_num.columns:
    plt.hist(df_num[i])
    plt.title(i)
    plt.show()

print(df_num.corr())
sns.heatmap(df_num.corr())

#Comparing survival rate across Age, SibSp, Parch and Fare
pd.pivot_table(training, index = 'Survived', values = ['Age','SibSp','Parch','Fare'])

for i in df_cat.columns:
    sns.barplot(df_cat[i].value_counts().index,df_cat[i].value_counts()).set_title(i)
    plt.show()

##Considering feature engineering because of the cabin and ticket graphs
#Comparing survival with the presented cattegorical values
print(pd.pivot_table(training, index = 'Survived' , columns = 'Pclass' , values = 'Ticket',  aggfunc = 'count'))
print()
print(pd.pivot_table(training, index = 'Survived', columns = 'Sex', values = 'Ticket', aggfunc = 'count'))
print()
print(pd.pivot_table(training, index = 'Survived', columns = 'Embarked', values = 'Ticket', aggfunc = 'count'))
print()

#Feature engineering
#Does the number of spouses/siblings and parents/child affect the males susceptibility to death by drowning because of having to save them?
#Build new feature that adds value of Sibsp and Parch called relatives
df_num.Parch
training['Relatives'] = training['Parch'] + training ['SibSp']
print(training.head())
Value = pd.loc['Relatives'].count() 
   