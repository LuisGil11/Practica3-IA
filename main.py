import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.api.layers import SimpleRNN, Dense, LSTM
from keras.src.models import Sequential
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# Cargar los datos
daily_data = pd.read_csv('./SN_d_tot_V2.0.csv', delimiter=";", header=0)
daily_df = daily_data[['DecimalYear', 'SN']]
daily_df = daily_df[daily_df["SN"] != -1]

# Dividir los datos en conjunto de entrenamiento y prueba
train_df = daily_df[daily_df['DecimalYear'] < 2014]
test_df = daily_df[daily_df['DecimalYear'] >= 2014]

# Escalar los datos
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_df[['DecimalYear', 'SN']])
test_scaled = scaler.transform(test_df[['DecimalYear', 'SN']])

# Preparar los datos para la RNN
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 1])  # Usar la columna 'SN' para las secuencias
        Y.append(data[i + time_step, 1])
    return np.array(X), np.array(Y)

time_step = 10
X_train, y_train = create_dataset(train_scaled, time_step)
X_test, y_test = create_dataset(test_scaled, time_step)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print(X_test)

# Construir el modelo
model = Sequential()
model.add(SimpleRNN(50, input_shape=(time_step, 1), activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluar el modelo
loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Model Loss: {loss}')

# Predecir y desescalar los valores
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(np.concatenate((X_train[:, -1, :], train_predict), axis=1))[:, 1]
test_predict = scaler.inverse_transform(np.concatenate((X_test[:, -1, :], test_predict), axis=1))[:, 1]

# Graficar los resultados
plt.figure(figsize=(14, 7))
plt.plot(daily_df['DecimalYear'], daily_df['SN'], label='Original Data')
plt.plot(train_df['DecimalYear'][time_step:], train_predict, label='Train Predict')
plt.plot(test_df['DecimalYear'][time_step:], test_predict, label='Test Predict')
plt.xlabel('Fecha')
plt.ylabel('Número de manchas solares')
plt.title('Predicción de manchas solares con RNN')
plt.legend()
plt.show()

# Graficar el error cuadrático medio para cada época
plt.plot(history.history['loss'], label='Error cuadrático medio en entrenamiento')
plt.plot(history.history['val_loss'], label='Error cuadrático medio en validación')
plt.xlabel('Épocas')
plt.ylabel('Error cuadrático medio')
plt.title('Evolución del Error Cuadrático Medio durante el Entrenamiento')
plt.legend()
plt.show()