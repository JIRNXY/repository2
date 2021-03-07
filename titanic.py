import pandas as pd
##import numpy as np
from pandas import Series,DataFrame
from setMissingAges import setMissingAges
from featureEngineering import featureEngineering
from logisticRegression import logisticRegression
from logisticRegressionBagging import logisticRegressionBagging
from aNN import aNN

from crossValidation import crossValidation
#first
data_train = pd.read_csv("D:/JIR/File/Python3.7.7/titanic/Train.csv")
##data_train.info()
data_test = pd.read_csv("D:/JIR/File/Python3.7.7/titanic/Test.csv")
##data_test.info()

#third 
data_train = setMissingAges(data_train)
data_test.loc[(data_test.Fare.isnull()), 'Fare'] = data_test['Fare'].mean();
data_test = setMissingAges(data_test)
##data_train['Age'] = data_train['Age'].fillna(data_train['Age'].mean());
##data_test['Age'] = data_test['Age'].fillna(data_test['Age'].mean());

#fourth
data_train = featureEngineering(data_train)
data_test = featureEngineering(data_test)
##shuffle_data_train = data_train.sample(frac=1)
##shuffle_data_test = data_test.sample(frac=1)

###fifth
regex1 = 'Survived|Age_.*|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*|Mother|Child|Family|Status'
regex2 = 'Age_.*|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*|Mother|Child|Family|Status'
##regex1 = 'Survived|Age_.*|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*'
##regex2 = 'Age_.*|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*'
train_df, model = logisticRegression(data_train,data_test,regex1,regex2)
##train_df, model = logisticRegressionBagging(data_train,data_test,regex1,regex2)
##train_df, model, pre = aNN(data_train,data_test,regex1,regex2)

##sixth
##coef = pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(model.coef_.T)})
bad = crossValidation(data_train,regex1,model)
