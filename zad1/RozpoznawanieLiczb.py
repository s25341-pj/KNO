import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from numpy import argmax

def main():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    if tf.io.gfile.exists("model.keras"):
        model = tf.keras.models.load_model(
            "model.keras"
        )
    else:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=5)

        model.save("model.keras")

    e = model.evaluate(x_test, y_test)
    print(e)

    def load_image(plik):
        img = Image.open(plik).convert('L')
        img = img.resize((28, 28))
        img = ImageOps.invert(img)
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = img_array.reshape((1, 28, 28))
        return img_array

    img_path = 'osiem.png'
    new_image = load_image(img_path)
    pred = model.predict(new_image)
    print(f"Liczba: {argmax(pred)}")

if __name__ == '__main__':
    main()