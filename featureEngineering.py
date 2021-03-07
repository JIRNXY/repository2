import sklearn.preprocessing as sp
import pandas as pd
#fourth
#特征提取Embarked,Sex,Pclass
#对Fare和Age特征缩放，梯度下降更快收敛
def featureEngineering(df):
    dummiesEmbarked = pd.get_dummies(df['Embarked'], prefix= 'Embarked')
    dummiesSex = pd.get_dummies(df['Sex'], prefix= 'Sex')
    dummiesPclass = pd.get_dummies(df['Pclass'], prefix= 'Pclass')
    df['Mother'] = df['Name'].replace({'.*Mrs.*': 1,'.*': 0}, regex=True)
    df['Child'] = df['Age']
    df.loc[(df.Child<=12), 'Child'] = 1;
    df.loc[(df.Child>12), 'Child'] = 0;
    df['Family'] = df['SibSp'] + df['Parch']
    df['Status'] = df['Name'].replace({'.*(Master|Dr|Rev|Capt).*': 1,'.*': 0}, regex=True)
    
    df = pd.concat([df, dummiesEmbarked, dummiesSex, dummiesPclass], axis=1)
    df.drop([  'Pclass','Name','SibSp','Parch','Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    
    scaler = sp.StandardScaler()
    ageScaleParam = scaler.fit(df[['Age']])
    df[['Age_scaled']] = scaler.fit_transform(df[['Age']], ageScaleParam)
    fareScaleParam = scaler.fit(df[['Fare']])
    df[['Fare_scaled']] = scaler.fit_transform(df[['Fare']], fareScaleParam)
    return df
