import os
import pickle

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input, Conv2D, ReLU, Flatten, \
    Dense, Reshape, Conv2DTranspose, Activation, Lambda
from tensorflow.python.keras import backend as K
import numpy as np

from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.losses import MeanSquaredError

from keras.layers.normalization.batch_normalization import BatchNormalization
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

class VAE:
    """
    Autoencoder represents a Deep Convolutional autoencoder architecture with
    mirrored encoder and decoder components.
    """

    def __init__(self,
                 input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_space_dim):

        # Spectograms saranno trattati come grayscale images
        self.input_shape = input_shape # ex. [28, 28, 1] (width, height, # channels)

        # indica il numero di filters per ogni layer
        self.conv_filters = conv_filters # ex. [2, 4, 8] --> il primo layer avrà 2 filtri, il secondo 4, etc...

        # indica il tipo di kernel per ogni layer
        self.conv_kernels = conv_kernels # ex. [3, 5, 3] --> il primo layer avrà kernel 3X3, il secondo 5X5, etc...

        # indica lo stride usato in ogni layer
        self.conv_strides = conv_strides # [1, 2, 2] --> lo stride per il primo layer sarà 1, quello per il secondo sarà 2, etc...

        self.latent_space_dim = latent_space_dim # ex. 2 (2 dimensioni)

        self.encoder = None
        self.decoder = None
        self.model = None

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None

        self._build()



    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()


    def compile(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss=self._calculate_combined_loss,
                           metrics=[self._calculate_reconstruction_loss,
                                    self._calculate_kl_loss])


    def train(self, x_train, batch_size, num_epochs):
        self.model.fit(x_train,
                       x_train,
                       batch_size=batch_size,
                       epochs=num_epochs,
                       shuffle=True)

    # Save autoencoder
    def save(self, save_folder="."):
        self._create_folder_if_it_doesnt_exists(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)



    def _create_folder_if_it_doesnt_exists(self, save_folder):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)


    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]

        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)


    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights.h5")
        self.model.save_weights(save_path)


    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)


    def reconstruct(self, images):
        latent_representations = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_representations)

        return reconstructed_images, latent_representations



    @classmethod
    def load(cls, save_folder="."):

        # load parameters
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)

        autoencoder = VAE(*parameters)

        # load weights
        weights_path = os.path.join(save_folder, "weights.h5")
        autoencoder.load_weights(weights_path)

        return autoencoder

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()


    # autoencoder
    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name="autoencoder")


    # Encoder
    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name="encoder")


    def _add_encoder_input(self):

        # creiamo l'input layer
        return Input(shape=self.input_shape, name="encoder_input")


    def _add_conv_layers(self, encoder_input):
        """Creates convolutional blocks in encoder."""

        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)

        return x


    def _add_conv_layer(self, layer_index, x):
        """Adds a convolutional block to a graph of layers, consisting of
        conv 2d + ReLU activation + batch normalization.
        """
        layer_number = layer_index + 1
        conv_layer = Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"enconder_conv_layer_{layer_number}"
        )
        x = conv_layer(x)
        x = ReLU(name=f"encoder_relu_{layer_number}")(x)
        x = BatchNormalization(name=f"encoder_bn_{layer_number}")(x)

        return x


    def _add_bottleneck(self, x):
        """Flatten data and add bottleneck with Guassian sampling (Dense
        layer).
        """
        self._shape_before_bottleneck = K.int_shape(x)[1:]
        x = Flatten()(x)
        self.mu = Dense(self.latent_space_dim, name="mu")(x)
        self.log_variance = Dense(self.latent_space_dim,
                                  name="log_variance")(x)

        def sample_point_from_normal_distribution(args):
            mu, log_variance = args
            epsilon = K.random_normal(shape=K.shape(self.mu), mean=0.,
                                      stddev=1.)
            sampled_point = mu + K.exp(log_variance / 2) * epsilon
            return sampled_point

        x = Lambda(sample_point_from_normal_distribution,
                   name="encoder_output")([self.mu, self.log_variance])
        return x


    # decoder
    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layers = self._add_dense_layers(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layers)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")


    def _add_decoder_input(self):
        return Input(shape=self.latent_space_dim, name = "decoder_input")


    def _add_dense_layers(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck) # ex. _shape_before_bottleneck= [1, 2, 4] --> vogliamo il flatten 1 X 2 X 4 = 8
        dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input)
        return dense_layer


    def _add_reshape_layer(self, dense_layer):
        return Reshape(self._shape_before_bottleneck)(dense_layer)


    def _add_conv_transpose_layers(self, x):
        """Add conv transpose blocks."""
        # loop through all the conv layers in revere order and stop at the first layer
        for layer_index in reversed(range(1, self._num_conv_layers)): # ex. [0, 1, 2] -> [2, 1]
            x = self._add_conv_transpose_layer(layer_index, x)

        return x


    def _add_conv_transpose_layer(self, layer_index, x):
        layer_num = self._num_conv_layers - layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"decoder_conv_transpose_layer_{layer_num}"
        )
        x = conv_transpose_layer(x)
        x = ReLU(name=f"decoder_relu_{layer_num}")(x)
        x = BatchNormalization(name=f"decoder_bn_{layer_num}")(x)
        return x


    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(
            filters=1,
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name=f"decoder_conv_transpose_layer_{self._num_conv_layers}"
        )
        x = conv_transpose_layer(x)
        output_layer = Activation("sigmoid", name="sigmoid_layer")(x)
        return output_layer


    def _calculate_reconstruction_loss(self, y_target, y_predicted):
        error = y_target - y_predicted
        reconstruction_loss = K.mean(K.square(error), axis=[1, 2, 3])
        return reconstruction_loss


    def _calculate_kl_loss(self, y_target, y_predicted):
        kl_loss = -0.5 * K.sum(1 + self.log_variance - K.square(self.mu) -
                               K.exp(self.log_variance), axis=1)
        return kl_loss


    def _calculate_combined_loss(self, y_target, y_predicted):
        reconstruction_loss = self._calculate_reconstruction_loss(y_target, y_predicted)
        kl_loss = self._calculate_kl_loss(y_target, y_predicted)
        combined_loss = self.reconstruction_loss_weight * reconstruction_loss\
                                                         + kl_loss
        return combined_loss


if __name__ == "__main__":
    autoencoder = VAE(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2
    )
    autoencoder.summary()