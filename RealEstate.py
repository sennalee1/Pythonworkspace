import pandas as pd

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
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax'),

])

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit( np.array(x데이터), np.array(y데이터), epochs=10 )