import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from keras_tuner.tuners import Hyperband


def load_and_prepare_data(file_path):
    columns = [
        'Class',
        'Alcohol',
        'Malic_acid',
        'Ash',
        'Alcalinity_of_ash',
        'Magnesium',
        'Total_phenols',
        'Flavanoids',
        'Nonflavanoid_phenols',
        'Proanthocyanins',
        'Color_intensity',
        'Hue',
        'OD280/OD315',
        'Proline'
    ]
    wine_data = pd.read_csv(file_path, header=None, names=columns)
    wine_data = wine_data.sample(frac=1, random_state=42).reset_index(drop=True)


    classes = wine_data[['Class']]
    features = wine_data.drop(columns=['Class'])
    encoder = OneHotEncoder(sparse_output=False)
    classes_one_hot = encoder.fit_transform(classes)


    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)


    X_train, X_temp, y_train, y_temp = train_test_split(features_scaled, classes_one_hot, test_size=0.4, random_state=7)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=7)

    #return features_scaled, classes_one_hot
    return X_train, X_val, X_test, y_train, y_val, y_test


class WineClassifier(Model):
    def __init__(self, input_dim, hidden_units, output_dim):
        super(WineClassifier, self).__init__()
        self.hidden_layers = [layers.Dense(units, activation='relu') for units in hidden_units]
        self.output_layer = layers.Dense(output_dim, activation='softmax')

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

    def build(self, input_shape):
        super(WineClassifier, self).build(input_shape)


def build_model(hp):
    hidden_units = [
        hp.Int(f'units_layer_{i}', min_value=1, max_value=64, step=5)
        for i in range(hp.Int('num_layers', 1, 3))
    ]
    learning_rate = hp.Choice('learning_rate', [0.01, 0.02, 0.03])
    model = WineClassifier(input_dim=13, hidden_units=hidden_units, output_dim=3)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


if __name__ == "__main__":

    file_path = 'wine.data'
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data(file_path)


    tuner = Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=50,
        factor=3,
        directory='tuner',
        project_name='wine_classifier'
    )


    tuner.search(
        X_train, y_train,
        epochs=100,
        validation_data=(X_val, y_val),
    )


    best_hps = tuner.get_best_hyperparameters(1)[0]
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=1,
    )


    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")


    model.save("optimized_wine_model.keras")