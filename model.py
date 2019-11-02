from tensorflow import keras

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28, 1)), #warstwy sieci neuronowej, jej architektura
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(62, activation='softmax')
])