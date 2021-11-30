import pandas as pd
import matplotlib as plt
import sklearn


data = pd.read_csv('richeville2.csv')


# print(data['value']) 해당 열만 출력, data['value'].min() 최소값 출력, max는 최대값 출력.

y데이터 = data['value'].values # 리스트로 다 담아줌


x데이터 = []

for i, rows in data.iterrows():   # iterrow - dataframe을 한 행씩 출력
    x데이터.append( [ rows['size'], rows['family number'], rows['remodeling'], rows['year'],
    rows['thermal'], rows['ele'], rows['high'], rows['Highway'], rows['subway'],
    rows['Centraltime'], rows['Centralmeter'], rows['Bigmart'], rows['hospital'],
    rows['park'], rows['peopledst'], rows['workerdst'] ] )



import numpy as np
import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(1)

])

model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

model.fit( np.array(x데이터), np.array(y데이터), epochs=100 )

# 예측
# 예측값 = model.predict ( [ [23.0], [1058.0], [62.0], [5.0], [0.0], [0.0], [0.0],
#  [2300.0], [3800.0], [36.0], [15100.0], [795.0], [10600.0], [261.0], [89.9], [16.7] ] )
# print(예측값)

xhat = ( [ [23.0], [1058.0], [62.0], [5.0], [0.0], [0.0], [0.0],
 [2300.0], [3800.0], [36.0], [15100.0], [795.0], [10600.0], [261.0], [89.9], [16.7] ] )
yhat = model.predict(xhat)

plt.figure()
plt.plot(yhat, label = "predicted")
plt.plot(y데이터, label = "actual")

plt.legend(prop={'size':20})

print("Evaluate : {}".format(np.average(np.sqrt((yhat - y데이터)**2))))