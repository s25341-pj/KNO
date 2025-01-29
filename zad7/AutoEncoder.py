import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image_dataset_from_directory


image_size = (128, 128)
batch_size = 16
latent_dim = 2
epochs = 200

dataset = image_dataset_from_directory(
    "picture",
    label_mode=None,
    image_size=image_size,
    batch_size=batch_size
)


dataset = dataset.map(lambda x: (x, x))
dataset = dataset.map(lambda x, y: (x / 255.0, y / 255.0))


data_argmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
])


def augment_images(x, y):
    x = data_argmentation(x)
    return x, y


augmented_dataset = dataset.map(augment_images)


encoder_input = layers.Input(shape=(128, 128, 3))
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(encoder_input)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Flatten()(x)
latent_output = layers.Dense(latent_dim, activation="relu")(x)

encoder = models.Model(encoder_input, latent_output, name="encoder")

decoder_input = layers.Input(shape=(latent_dim,))
x = layers.Dense(32 * 32 * 64, activation="relu")(decoder_input)
x = layers.Reshape((32, 32, 64))(x)
x = layers.Conv2DTranspose(64, (3, 3), activation="relu", padding="same")(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2DTranspose(32, (3, 3), activation="relu", padding="same")(x)
x = layers.UpSampling2D((2, 2))(x)
decoder_output = layers.Conv2DTranspose(3, (3, 3), activation="sigmoid", padding="same")(x)

decoder = models.Model(decoder_input, decoder_output, name="decoder")

autoencoder_input = encoder_input
autoencoder_output = decoder(encoder(autoencoder_input))
autoencoder = models.Model(autoencoder_input, autoencoder_output, name="autoencoder")

autoencoder.compile(optimizer="adam", loss="mse")

history = autoencoder.fit(
    augmented_dataset,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True
)



def reconstruct_images(autoencoder, dataset, num_images=5):
    for images in dataset.take(1):
        reconstructed_images = autoencoder.predict(images)

        plt.figure(figsize=(10, 5))
        for i in range(num_images):
            plt.subplot(2, num_images, i + 1)
            plt.imshow(images[i])
            plt.axis("off")
            plt.title("Original")

            plt.subplot(2, num_images, i + 1 + num_images)
            plt.imshow(reconstructed_images[i])
            plt.axis("off")
            plt.title("Reconstructed")
        plt.suptitle("Original vs Reconstructed Images")
        plt.show()



def generate_new_images(decoder, num_images):

    random_latent_vectors = np.random.normal(size=(num_images, latent_dim))

    generated_images = decoder.predict(random_latent_vectors)

    plt.figure(figsize=(10, 2))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(generated_images[i])
        plt.axis("off")
    plt.suptitle("Generated Images")
    plt.show()


reconstruct_images(autoencoder, augmented_dataset, num_images=5)
generate_new_images(decoder, num_images=1)
