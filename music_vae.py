import numpy as np
import tensorflow as tf
import keras
from keras import layers
import os
import pickle
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

# this sampling layer is the bottleneck layer of variational autoencoder,
# it uses the output from two dense layers z_mean and z_log_var as input, 
# convert them into normal distribution and pass them to the decoder layer
 
class Sampling(layers.Layer):
    """Uses (mean, log_var) to sample z, the vector encoding a digit."""
 
    def call(self, inputs):
        mean, log_var = inputs
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return mean + tf.exp(0.5 * log_var) * epsilon
    


class VAE(keras.Model):
    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides,
                 latent_space_dim, **kwargs):
        super().__init__(**kwargs)
        self.input_shape = input_shape # [28, 28, 1]
        self.conv_filters = conv_filters # [2, 4, 8]
        self.conv_kernels = conv_kernels # [3, 5, 3]
        self.conv_strides = conv_strides # [1, 2, 2]
        self.latent_space_dim = latent_space_dim # 2
        self.reconstruction_loss_weight = 1000000

        self.encoder = None
        self.decoder = None

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None

        self._build_model()

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
 
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
 
    def train_step(self, data):
        with tf.GradientTape() as tape:
            mean,log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2),
                )
            )
            kl_loss = -0.5 * (1 + log_var - tf.square(mean) - tf.exp(log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def reconstruct(self, images):
        latent_representations = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images, latent_representations
    
    def _build_model(self):
        ## Build encoder
        self._model_input = keras.Input(shape=self.input_shape) 
        # add conv layers 
        x = self._model_input
        for layer_index in range(self._num_conv_layers):
            x = layers.Conv2D(filters=self.conv_filters[layer_index],
                                kernel_size=self.conv_kernels[layer_index],
                                activation="relu",
                                strides=self.conv_strides[layer_index],
                                padding="same")(x)
            x = layers.BatchNormalization()(x)
        # create bottleneck
        self._shape_before_bottleneck = K.int_shape(x)[1:]
        x = layers.Flatten()(x)
        x = layers.Dense(16, activation="relu")(x)
        mean = layers.Dense(self.latent_space_dim, name="mean")(x)
        log_var = layers.Dense(self.latent_space_dim, name="log_var")(x)
        z = Sampling()([mean, log_var])
        
        self.encoder = keras.Model(self._model_input, [mean, log_var, z], name="encoder")
        self.encoder.summary()

        ## Build decoder
        latent_inputs = keras.Input(shape=(self.latent_space_dim,))
        num_neurons = np.prod(self._shape_before_bottleneck)
        x = layers.Dense(num_neurons, activation="relu")(latent_inputs) #64 * 16 * 64
        x = layers.Reshape(self._shape_before_bottleneck)(x) # use shape before bottleneck
        # add conv transpose layers
        for layer_index in reversed(range(1, self._num_conv_layers)):
            x = layers.Conv2DTranspose(filters=self.conv_filters[layer_index],
                                        kernel_size=self.conv_kernels[layer_index],
                                        activation="relu",
                                        strides=self.conv_strides[layer_index],
                                        padding="same")(x)
            x = layers.BatchNormalization()(x)
        decoder_outputs = layers.Conv2DTranspose(1, 
                                                kernel_size=self.conv_kernels[0],
                                                activation="sigmoid", 
                                                strides=self.conv_strides[0], 
                                                padding="same")(x)
        self.decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        self.decoder.summary()

    @classmethod
    def load(cls, save_folder="model_vae"):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        print(parameters)
        autoencoder = VAE(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        autoencoder.load_weights(weights_path)
        return autoencoder


def load_fsdd(spectograms_path):
    x_train = []
    for root, _, file_names in os.walk(spectograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectogram = np.load(file_path) # ex. (n_bins, n_frames) --> abbiamo bisogno di 3 dimensioni (3 channels)-> (n_bins, n_frames, 1)
            x_train.append(spectogram)

    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis] # -> (3000, 256, 64, 1)
    #x_train = tf.data.Dataset.from_tensor_slices(x_train)
    return x_train


def plot_latent_space(vae, n=10, figsize=5):
    # display a n*n 2D manifold of images
    img_size = 28
    scale = 0.5
    figure = np.zeros((img_size * n, img_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of images classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]
 
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(sample, verbose=0)
            images = x_decoded[0].reshape(img_size, img_size)
            figure[
                i * img_size : (i + 1) * img_size,
                j * img_size : (j + 1) * img_size,
            ] = images
 
    plt.figure(figsize=(figsize, figsize))
    start_range = img_size // 2
    end_range = n * img_size + start_range
    pixel_range = np.arange(start_range, end_range, img_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()




if __name__=="__main__":
    tf.config.run_functions_eagerly(True)

    latent_dim = 2
 
    encoder_inputs = keras.Input(shape=(256, 64, 1)) #keras.Input(shape=(28, 28, 1)) #keras.Input(shape=(256, 64, 1)) 
    x = layers.Conv2D(32, 3, activation="relu", strides=1, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(64, 3, activation="relu", strides=1, padding="same")(x)
    # store shape before bottleneck
    shape_before_bottleneck = K.int_shape(x)[1:]
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    mean = layers.Dense(latent_dim, name="mean")(x)
    log_var = layers.Dense(latent_dim, name="log_var")(x)
    z = Sampling()([mean, log_var])
    encoder = keras.Model(encoder_inputs, [mean, log_var, z], name="encoder")
    encoder.summary()

    #TODO: add BatchNormalization layers 
    latent_inputs = keras.Input(shape=(latent_dim,))
    num_neurons = np.prod(shape_before_bottleneck)
    x = layers.Dense(num_neurons, activation="relu")(latent_inputs) #64 * 16 * 64
    x = layers.Reshape(shape_before_bottleneck)(x) # use shape before bottleneck
    
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    #x = layers.Conv2DTranspose(32, 3, activation="relu", strides=1, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", strides=1, padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()



    # mnist input
    #(x_train, _), (x_test, _) = keras.datasets.fashion_mnist.load_data()
    #fashion_mnist = np.concatenate([x_train, x_test], axis=0)
    #fashion_mnist = np.expand_dims(fashion_mnist, -1).astype("float32") / 255
    
    print("before loading spectrograms")

    # spectrogram input
    SPECTOGRAMS_PATH = os.path.join("dataset", "fsdd", "spectrograms")
    x_train = load_fsdd(SPECTOGRAMS_PATH)

    print("spectrograms loaded")

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())
    #vae.fit(fashion_mnist, epochs=10, batch_size=128)
    vae.fit(x_train, epochs=10, batch_size=32)

    WEIGHTS_Path = os.path.join("model", "trial.weights.h5")
    vae.save_weights(WEIGHTS_Path)
    #vae.load_weights(os.path.join('model_vae', 'weights.h5'))

    print("weights loaded now")

    # plotting the latent space
    plot_latent_space(vae)