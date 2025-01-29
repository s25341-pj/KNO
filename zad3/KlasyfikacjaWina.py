import pandas as pd
import tensorflow as tf
import sklearn.model_selection as ms
import datetime
import numpy as np
import json
import random
import itertools


SEED = 30
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


wineData = pd.read_csv('wine.data')
wineData.columns = [
    'Class',
    'Alcohol',
    'Malic acid',
    'Ash',
    'Alcalinity of ash',
    'Magnesium',
    'Total phenols',
    'Flavanoids',
    'Nonflavanoid phenols',
    'Proanthocyanins',
    'Color intensity',
    'Hue',
    'OD280/OD315 of diluted wines',
    'Proline'
]


wineData1 = pd.get_dummies(wineData, columns=['Class'], dtype=float)


wineData['Class'] = wineData['Class'].apply(lambda x: x-1)
Y = wineData['Class'].values
X = wineData.drop('Class', axis=1).values


X_train, X_temp, Y_train, Y_temp = ms.train_test_split(X, Y, test_size=0.4, random_state=SEED)
X_val, X_test, Y_val, Y_test = ms.train_test_split(X_temp, Y_temp, test_size=0.5, random_state=SEED)


def create_model(learning_rate, layers_config, dropout_rate):

    model = tf.keras.Sequential()
    for units in layers_config:
        model.add(tf.keras.layers.Dense(units, activation='relu'))
    if dropout_rate > 0:
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

learning_rates = [0.001, 0.01]
layer_configs = [[64, 32], [128, 64, 32]]
dropout_rates = [0.0, 0.5]
param_combinations = list(itertools.product(learning_rates, layer_configs, dropout_rates))


results = []
trained_models = []

for lr, layers, dropout in param_combinations:
    print(f"Trening dla kombinacji: LR={lr}, Layers={layers}, Dropout={dropout}")
    model = create_model(learning_rate=lr, layers_config=layers, dropout_rate=dropout)
    history = model.fit(
        X_train, Y_train,
        epochs=150,
        batch_size=32,
        validation_data=(X_val, Y_val),
        verbose=0
    )
    val_loss = history.history['val_loss'][-1]
    val_accuracy = history.history['val_accuracy'][-1]

    results.append({
        "Learning_Rate": lr,
        "Layers_Config": layers,
        "Dropout_Rate": dropout,
        "Validation_Loss": val_loss,
        "Validation_Accuracy": val_accuracy
    })
    trained_models.append(model)


best_index = max(range(len(results)), key=lambda i: results[i]['Validation_Accuracy'])
best_model = trained_models[best_index]  # Pobranie najlepszego modelu
best_result = results[best_index]  # Informacje o najlepszym modelu

# Model 1
model1 = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],), name='Layer_1'),
    tf.keras.layers.Dense(32, activation='relu', name='Layer_2'),
    tf.keras.layers.Dense(3, activation='softmax', name='Output_Layer')
])

model1.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model1.summary()


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history1 = model1.fit(
    X_train, Y_train,
    epochs=250,
    batch_size=100,
    callbacks=[tensorboard_callback],
    validation_data=(X_val, Y_val),
    verbose=0
)

EV1 = model1.evaluate(X_test, Y_test)

# Model 2
model2 = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],), name='Layer_1'),
    tf.keras.layers.Dense(64, activation='relu', name='Layer_2'),
    tf.keras.layers.Dropout(0.5, name='Dropout_Layer'),
    tf.keras.layers.Dense(32, activation='relu', name='Layer_3'),
    tf.keras.layers.Dense(16, activation='softmax', name='Layer_4'),
    tf.keras.layers.Dense(3, activation='softmax', name='Output_Layer')
])

model2.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model2.summary()

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback2 = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history2 = model2.fit(
    X_train, Y_train,
    epochs=150,
    batch_size=100,
    callbacks=[tensorboard_callback2],
    validation_data=(X_val, Y_val),
    verbose=0
)

EV2 = model2.evaluate(X_test, Y_test)


baseline_file = "baseline_results.json"

baseline_data = {
    "Model_1": {
        "Validation_Loss": history1.history['val_loss'][-1],
        "Validation_Accuracy": history1.history['val_accuracy'][-1],
        "Test_Loss": EV1[0],
        "Test_Accuracy": EV1[1]
    },
    "Model_2": {
        "Validation_Loss": history2.history['val_loss'][-1],
        "Validation_Accuracy": history2.history['val_accuracy'][-1],
        "Test_Loss": EV2[0],
        "Test_Accuracy": EV2[1]
    }
}

with open(baseline_file, "w") as file:
    json.dump(baseline_data, file, indent=4)

print(f"Wyniki bazowe zapisane do pliku {baseline_file}")


results_df = pd.DataFrame(results)


results_df.to_csv("model_results.csv", index=False)


best_result = results_df.loc[results_df['Validation_Accuracy'].idxmax()]

print("Najlepszy model:")
print(best_result)

print(type(best_model))


log_dir_best_model = "logs/fit_best_model/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback_best_model = tf.keras.callbacks.TensorBoard(log_dir=log_dir_best_model, histogram_freq=1)


history_best_model = best_model.fit(
    X_train, Y_train,
    epochs=150,  # Liczba epok do trenowania
    batch_size=32,
    validation_data=(X_val, Y_val),
    callbacks=[tensorboard_callback_best_model],
    verbose=1
)


EV_best_model = best_model.evaluate(X_test, Y_test)

print("Wyniki najlepszego modelu na zbiorze testowym:")
print(f"Test Loss: {EV_best_model[0]}")
print(f"Test Accuracy: {EV_best_model[1]}")

with open(baseline_file, "r") as file:
    baseline_data = json.load(file)


baseline_model1 = baseline_data["Model_1"]
baseline_model2 = baseline_data["Model_2"]


best_model_results = best_result.to_dict()


print("\nPorównanie wyników modeli:")
print(f"Model 1 (baseline) - Validation Accuracy: {baseline_model1['Validation_Accuracy']}, Test Accuracy: {baseline_model1['Test_Accuracy']}")
print(f"Model 2 (baseline) - Validation Accuracy: {baseline_model2['Validation_Accuracy']}, Test Accuracy: {baseline_model2['Test_Accuracy']}")
print(f"Najlepszy model - Validation Accuracy: {best_model_results['Validation_Accuracy']}")


best_test_accuracy = best_model_results['Validation_Accuracy']
print(f"Najlepszy model - Test Accuracy: {best_test_accuracy}")


if best_test_accuracy > baseline_model1['Test_Accuracy'] and best_test_accuracy > baseline_model2['Test_Accuracy']:
    print("\nNajlepszy model jest lepszy niż modele baseline pod względem dokładności testowej.")
else:
    print("\nModele baseline są lepsze niż najlepszy model w badaniu.")


# def predict_wine_class(features, model):
#     features = np.array(features).reshape(1, -1)
#     prediction = model.predict(features)
#     predicted_class = np.argmax(prediction)
#     return predicted_class
#
# features = [12.5, 2.5, 2.8, 20.0, 120.0, 2.8, 2.69, 0.34, 1.56, 6.1, 1.04, 3.0, 1.1]
#
# predicted_class = predict_wine_class(features, model1)
# print(f"Kategoria wina to: {predicted_class + 1}")