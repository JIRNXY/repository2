from sklearn import linear_model
import pandas as pd
from sklearn import model_selection

def crossValidation(df,regex1,model):
    data_train = pd.read_csv("D:/JIR/File/Python3.7.7/titanic/Train.csv")
    split_train, split_cv = model_selection.train_test_split(df, test_size=0.3, random_state=0)

    train_df = split_train.filter(regex=regex1)
    y = train_df.values[:,0]
    x = train_df.values[:,1:]

    print (model_selection.cross_val_score(model, x, y, cv=5))

    cv_df = split_cv.filter(regex=regex1)
    predictions = model.predict(cv_df.values[:,1:])
    
    bad_cases = data_train.loc[data_train['PassengerId'].isin(split_cv[predictions != cv_df.values[:,0]]['PassengerId'].values)]
    return bad_cases
