import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler


t = np.linspace(0, 10 * np.pi, 1000)
y = np.sin(t)


plt.figure(figsize=(10, 4))
plt.plot(t, y, label='Funkcja sinus')
plt.legend()
plt.show()


scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

N_STEPS = 30
FUTURE_STEPS = 150


def create_sequences(data, n_steps):
    X, Y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        Y.append(data[i + n_steps])
    return np.array(X), np.array(Y)

X, Y = create_sequences(y_scaled, N_STEPS)
X = X.reshape((X.shape[0], X.shape[1], 1))

#LSTM
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(N_STEPS, 1)),
    LSTM(50, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')


EPOCHS = 20
model.fit(X, Y, epochs=EPOCHS, batch_size=16, verbose=1)



last_sequence = X[-1]
predictions = []


for _ in range(FUTURE_STEPS):
    next_value = model.predict(last_sequence.reshape(1, N_STEPS, 1), verbose=0)
    predictions.append(next_value[0, 0])
    last_sequence = np.roll(last_sequence, -1)
    last_sequence[-1] = next_value


predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))


extended_t = np.linspace(t[0], t[-1] + FUTURE_STEPS * (t[1] - t[0]), len(t) + FUTURE_STEPS)


plt.figure(figsize=(10, 4))
plt.plot(t, y, label='Oryginalna funkcja')
plt.plot(extended_t[-FUTURE_STEPS:], predictions, label='Przewidywana kontynuacja', linestyle='dashed')
plt.legend()
plt.show()
