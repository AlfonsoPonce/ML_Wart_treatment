import pandas as pd
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense

# Data Extraction
file_errors_location = './Cryotherapy.xlsx'
df = pd.read_excel(file_errors_location)
df = df.to_numpy()

X = df[:,0:5]
Y = df[:,6]

# Preprocessing
normalized_data = preprocessing.normalize(X)

input_shape = len(X[1])

# Create Network
model = Sequential()
model.add(Dense(12, input_shape=(input_shape,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=150, batch_size=10)

# Evaluate the keras model
_, accuracy = model.evaluate(X, Y)
print('Accuracy: %.2f' % (accuracy*100))




