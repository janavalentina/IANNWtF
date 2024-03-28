import os
import numpy as np
import tensorflow as tf

from vae import VAE


LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCHS = 150

SPECTOGRAMS_PATH = os.path.join("dataset", "fsdd", "spectrograms")

def load_fsdd(spectograms_path):
    x_train = []
    for root, _, file_names in os.walk(spectograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectogram = np.load(file_path) # ex. (n_bins, n_frames) --> abbiamo bisogno di 3 dimensioni (3 channels)-> (n_bins, n_frames, 1)
            x_train.append(spectogram)

    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis] # -> (3000, 256, 64, 1)
    x_train = tf.data.Dataset.from_tensor_slices((x_train, x_train))

    return x_train

@tf.function
def train(x_train, learning_rate, batch_size, epochs):
    autoencoder = VAE(
        input_shape=(256, 64, 1),
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2, (2, 1)),
        latent_space_dim=128
    )
    autoencoder.summary()
    #autoencoder.compile(learning_rate)
    #autoencoder.train(x_train, batch_size, epochs)
    autoencoder.compile(optimizer=tf.compat.v1.keras.optimizers.Adam(learning_rate=learning_rate))
    autoencoder.fit(x_train, epochs, batch_size, steps_per_epoch=len(x_train)//batch_size, shuffle=True)
    return autoencoder


if __name__ == "__main__":
    x_train = load_fsdd(SPECTOGRAMS_PATH)
    autoencoder = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder.save("model_vae")