from sklearn.ensemble import RandomForestRegressor
#second
#使用 RandomForestClassifier 填补缺失的年龄属性
def setMissingAges(df):
    
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    knownAge = age_df[age_df.Age.notnull()].values
    unknownAge = age_df[age_df.Age.isnull()].values

    y = knownAge[:, 0]

    x = knownAge[:, 1:]

    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(x, y)
    
    predictedAges = rfr.predict(unknownAge[:, 1::])
    
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 
    
    return df
