from sklearn import linear_model
import pandas as pd
import numpy as np
from plotLearningCurve import plotLearningCurve
#fifth
# logistic回归模型
def logisticRegression(df1,df2,regex1,regex2):
    data_test = pd.read_csv("D:/JIR/File/Python3.7.7/titanic/Test.csv")
    train_df = df1.filter(regex=regex1)
    train_np = train_df.values
    
    y = train_np[:, 0]
    x = train_np[:, 1:]
    
    llr = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6, solver='liblinear')
    llr.fit(x, y)

    test_df = df2.filter(regex=regex2)
    predictions = llr.predict(test_df)
    
    result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
    result.to_csv("D:/JIR/File/Python3.7.7/titanic/logistic_regression_predictions.csv", index=False)
    plotLearningCurve(llr, "学习曲线", x, y)
    return train_df,llr
