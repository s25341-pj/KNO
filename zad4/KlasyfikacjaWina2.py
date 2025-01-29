import datetime
import pandas as pd
import tensorflow as tf
import sklearn.model_selection as ms

def load_and_prepare_data(file_path):
    wine_data = pd.read_csv(file_path)
    wine_data.columns = [
        'Class',
        'Alcohol',
        'Malic acid',
        'Ash',
        'Alcalinity of ash  ',
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
    wine_data = wine_data.sample(frac=1, random_state=30).reset_index(drop=True)
    wine_data['Class'] = wine_data['Class'].apply(lambda x: x - 1)
    X = wine_data.drop('Class', axis=1).values
    Y = wine_data['Class'].values
    X_train, X_temp, Y_train, Y_temp = ms.train_test_split(X, Y, test_size=0.8, random_state=30)
    X_val, X_test, Y_val, Y_test = ms.train_test_split(X_temp, Y_temp, test_size=0.5, random_state=30)
    return X_train, X_val, X_test, Y_train, Y_val, Y_test

def create_simple_model(units_1, units_2, learning_rate):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units_1, activation='relu', input_shape=(13,), name='Layer_1'),
        tf.keras.layers.Dense(units_2, activation='relu', name='Layer_2'),
        tf.keras.layers.Dense(3, activation='softmax', name='Output_Layer')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def create_complex_model(units_1, units_2, units_3, learning_rate):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units_1, activation='relu', input_shape=(13,), name='Layer_1'),
        tf.keras.layers.Dense(units_2, activation='relu', name='Layer_2'),
        tf.keras.layers.Dense(units_3, activation='relu', name='Layer_3'),
        tf.keras.layers.Dense(3, activation='softmax', name='Output_Layer')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_evaluate_model(model, X_train, Y_train, X_val, Y_val, X_test, Y_test, log_dir):
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=100, batch_size=32, callbacks=[tensorboard_callback], verbose=1)
    validation_loss, validation_accuracy = model.evaluate(X_val, Y_val, verbose=0)
    test_loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=0)
    return validation_loss, validation_accuracy, test_loss, test_accuracy

def main():
    X_train, X_val, X_test, Y_train, Y_val, Y_test = load_and_prepare_data("wine.data")

    # Baseline model
    baseline_model = create_simple_model(64, 32, 0.01)
    baseline_log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_model_baseline"
    baseline_results = train_and_evaluate_model(baseline_model, X_train, Y_train, X_val, Y_val, X_test, Y_test, baseline_log_dir)

    baseline = {
        "model": "simple",
        "units_1": 64,
        "units_2": 32,
        "learning_rate": 0.01,
        "validation_loss": baseline_results[0],
        "validation_accuracy": baseline_results[1],
        "test_loss": baseline_results[2],
        "test_accuracy": baseline_results[3]
    }

    print("Baseline Model:")
    print(f"Validation Loss: {baseline['validation_loss']:.5f}")
    print(f"Validation Accuracy: {baseline['validation_accuracy']:.5f}")
    print(f"Test Loss: {baseline['test_loss']:.5f}")
    print(f"Test Accuracy: {baseline['test_accuracy']:.5f}")

    # Hyperparameter experiments
    experiments = [
        {"model": "simple", "units_1": 64, "units_2": 32, "learning_rate": 0.1},
        {"model": "simple", "units_1": 128, "units_2": 64, "learning_rate": 0.001},
        {"model": "complex", "units_1": 128, "units_2": 64, "units_3": 32, "learning_rate": 0.1},
        {"model": "complex", "units_1": 256, "units_2": 128, "units_3": 64, "learning_rate": 0.001}
    ]

    best_model = {
        "model": None,
        "units_1": None,
        "units_2": None,
        "units_3": None,
        "learning_rate": None,
        "val_accuracy": 0,
        "test_accuracy": 0
    }

    for experiment in experiments:
        if experiment["model"] == "simple":
            model = create_simple_model(experiment["units_1"], experiment["units_2"], experiment["learning_rate"])
        elif experiment["model"] == "complex":
            model = create_complex_model(experiment["units_1"], experiment["units_2"], experiment["units_3"], experiment["learning_rate"])

        log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_model_experiment"
        validation_loss, validation_accuracy, test_loss, test_accuracy = train_and_evaluate_model(model, X_train, Y_train, X_val, Y_val, X_test, Y_test, log_dir)

        if validation_accuracy > best_model["val_accuracy"]:
            best_model.update({
                "model": experiment["model"],
                "units_1": experiment["units_1"],
                "units_2": experiment["units_2"],
                "units_3": experiment.get("units_3"),
                "learning_rate": experiment["learning_rate"],
                "val_accuracy": validation_accuracy,
                "test_accuracy": test_accuracy
            })

    print("\nModel z poprzednich zajęć:")
    print(f"Model: {baseline['model']}")
    print(f"Units_1: {baseline['units_1']}")
    print(f"Units_2: {baseline['units_2']}")
    print(f"Learning Rate: {baseline['learning_rate']}")
    print(f"Validation Accuracy: {baseline['validation_accuracy']:.5f}")
    print(f"Test Accuracy: {baseline['test_accuracy']:.5f}")

    print("\nBest Model:")
    print(f"Model Type: {best_model['model']}")
    print(f"Units_1: {best_model['units_1']}")
    print(f"Units_2: {best_model['units_2']}")
    print(f"Units_3: {best_model['units_3']}")
    print(f"Learning Rate: {best_model['learning_rate']}")
    print(f"Validation Accuracy: {best_model['val_accuracy']:.5f}")
    print(f"Test Accuracy: {best_model['test_accuracy']:.5f}")

if __name__ == '__main__':
    main()