import librosa

from preprocess import MinMaxNormaliser

class SoundGenerator:
    """SoundGenerator is responsible for generating audio spectograms.
    """

    def __init__(self, vae, hop_length):
        self.vae = vae
        self.hop_length = hop_length
        self._min_max_normalizer = MinMaxNormaliser(0, 1)

    def generate(self, spectrograms, min_max_values):
        generated_spectrograms = [] 
        latent_representations = []

        for spectrogram in spectrograms:
            generated_spectogram, latent_representation = self.vae.reconstruct(spectrogram)
            generated_spectrograms.append(generated_spectogram)
            latent_representations.append(latent_representation)

        signals = self.convert_spectograms_to_audio(generated_spectrograms, min_max_values)
        return signals, latent_representations

    def convert_spectograms_to_audio(self, spectograms, min_max_values):
        signals = []
        for spectogram, min_max_value in zip(spectograms, min_max_values):
            # reshape the log spectogram
            log_spectogram = spectogram[:, :, 0] # elimina la dimensione 3

            # apply denormalization
            denorm_log_spec = self._min_max_normalizer.denormalise(log_spectogram,
                                                                   min_max_value["min"],
                                                                   min_max_value["max"])

            # log spectogram -> spectogram
            spec = librosa.db_to_amplitude(denorm_log_spec)

            # apply Griffin-Lim
            signal = librosa.istft(spec, hop_length=self.hop_length)

            # append signal to "signals"
            signals.append(signal)

        return signals
