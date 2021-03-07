import tensorflow as tf
import pandas as pd
import numpy as np

def aNN(df1, df2,regex1,regex2):
    data_test = pd.read_csv("D:/JIR/File/Python3.7.7/titanic/Test.csv")
    train_df = df1.filter(regex=regex1)
    train_np = train_df.values
    
    y = train_np[:, 0]
    x = train_np[:, 1:]

    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=32, input_dim=14, use_bias=True, kernel_initializer='uniform', bias_initializer='zeros', activation='relu'))
    model.add(tf.keras.layers.Dense(units=32, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Adam(0.003), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x, y)

    test_df = df2.filter(regex=regex2)
    predictions = model.predict(test_df)

    return train_df,model,predictions
