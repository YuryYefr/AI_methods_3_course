import numpy as np
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras import Sequential

# Генерація штучних даних
np.random.seed(42)
X = np.linspace(-1, 1, 100).reshape(-1, 1)  # Вхідні дані
y = X ** 3 + np.random.normal(0, 0.1, size=X.shape)  # Цільова функція з шумом

# Нормалізація даних
X = (X - X.mean()) / X.std()
y = (y - y.mean()) / y.std()


def mean_relative_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def feed_forward_model(hidden_neurons):
    model = Sequential([
        Dense(hidden_neurons, activation='relu', input_shape=(1,)),
        Dense(1)  # Вихідний шар
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return model


# a) 1 внутрішній шар з 10 нейронами
model_ff_1 = feed_forward_model(10)
model_ff_1.fit(X, y, epochs=100, verbose=0)
y_pred_ff_1 = model_ff_1.predict(X)
mre_ff_1 = mean_relative_error(y, y_pred_ff_1)

# b) 1 внутрішній шар з 20 нейронами
model_ff_2 = feed_forward_model(20)
model_ff_2.fit(X, y, epochs=100, verbose=0)
y_pred_ff_2 = model_ff_2.predict(X)
mre_ff_2 = mean_relative_error(y, y_pred_ff_2)


def cascade_forward_model(hidden_neurons, layers=1):
    model = Sequential()
    for _ in range(layers):
        model.add(Dense(hidden_neurons, activation='relu', input_shape=(1,)))
    model.add(Dense(1))  # Вихідний шар
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return model


# a) 1 внутрішній шар з 20 нейронами
model_cf_1 = cascade_forward_model(20)
model_cf_1.fit(X, y, epochs=100, verbose=0)
y_pred_cf_1 = model_cf_1.predict(X)
mre_cf_1 = mean_relative_error(y, y_pred_cf_1)

# b) 2 внутрішніх шари по 10 нейронів
model_cf_2 = cascade_forward_model(10, layers=2)
model_cf_2.fit(X, y, epochs=100, verbose=0)
y_pred_cf_2 = model_cf_2.predict(X)
mre_cf_2 = mean_relative_error(y, y_pred_cf_2)


def elman_model(neurons, layers=1):
    model = Sequential()
    for _ in range(layers):
        model.add(
            SimpleRNN(neurons, activation='relu', return_sequences=True if layers > 1 else False, input_shape=(1, 1)))
    model.add(Dense(1))  # Вихідний шар
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return model


# Перетворення X для RNN
X_rnn = X.reshape(X.shape[0], 1, 1)

# a) 1 внутрішній шар з 15 нейронами
model_el_1 = elman_model(15)
model_el_1.fit(X_rnn, y, epochs=100, verbose=0)
y_pred_el_1 = model_el_1.predict(X_rnn)
mre_el_1 = mean_relative_error(y, y_pred_el_1)

# b) 3 внутрішніх шари по 5 нейронів
model_el_2 = elman_model(5, layers=3)
model_el_2.fit(X_rnn, y, epochs=100, verbose=0)
y_pred_el_2 = model_el_2.predict(X_rnn)
mre_el_2 = mean_relative_error(y, y_pred_el_2)

results = {
    "Feed-Forward (10 нейронів)": mre_ff_1,
    "Feed-Forward (20 нейронів)": mre_ff_2,
    "Cascade-Forward (20 нейронів)": mre_cf_1,
    "Cascade-Forward (2x10 нейронів)": mre_cf_2,
    "Elman (15 нейронів)": mre_el_1,
    "Elman (3x5 нейронів)": mre_el_2,
}

# Вивід результатів
for model, error in results.items():
    print(f"{model}: MRE = {error:.2f}%")
