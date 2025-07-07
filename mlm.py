import numpy as np 
import pandas as pd
import tensorflow as tf
from keras.layers import Dense,Input
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

data=pd.read_csv('diabetes.csv')

x=data.drop('Outcome',axis=1)
y=data['Outcome']


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

tf.random.set_seed(42)
model = Sequential([
    Input(shape=(x_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy']
)

model.fit(
    x_train,y_train,epochs=50,batch_size=32
)

loss,accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

y_pred = (model.predict(x_test) >= 0.5).astype(int)
print(classification_report(y_test, y_pred))


model.save('diabetes_model.h5')

import joblib
joblib.dump(sc, 'scaler.pkl')
