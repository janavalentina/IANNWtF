import os
import librosa
import numpy as np
import pickle

# Free Spoken Digit Dataset: https://github.com/Jakobovski/free-spoken-digit-dataset


class Loader:
    """Loader is responssible for loading an audio file."""

    def __init__(self, sample_rate, duration, mono):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono

    def load(self, file_path):
        signal = librosa.load(file_path,
                              sr=self.sample_rate,
                              duration=self.duration,
                              mono=self.mono)[0]

        return signal



class Padder:
    """Padder is responsible to apply pafdding to an array."""

    def __init__(self, mode="constant"):
        self.mode = mode

    def left_pad(self, array, num_missing_items):
        # ex. [1, 2, 3], 2 -> [0, 0, 1, 2, 3]
        padded_array = np.pad(array,
                              (num_missing_items, 0),
                              mode=self.mode)
        return padded_array

    def right_pad(self, array, num_missing_items):
        # ex. [1, 2, 3], 2 -> [1, 2, 3, 0, 0]
        padded_array = np.pad(array,
                              (0, num_missing_items),
                              mode=self.mode)
        return padded_array


class LogSpectogramExtractor:
    """LogSpectogramExtractor extracts log spectorgrams (in dB) from a
    time-series signal"""

    def __init__(self, frame_size, hop_length):
        self.frame_size = frame_size
        self.hop_length = hop_length

    def extract(self, signal):
        stft = librosa.stft(signal,
                            n_fft=self.frame_size,
                            hop_length=self.hop_length)[:-1]
        spectogram = np.abs(stft)
        log_spectogram = librosa.amplitude_to_db(spectogram)
        return log_spectogram

class MinMaxNormaliser:
    """MinMaxNormaliser applies min max normalization to an array."""

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def normalise(self, array):
        norm_array = (array - array.min()) / (array.max() - array.min())
        norm_array = norm_array * (self.max_val - self.min_val) + self.min_val
        return norm_array

    def denormalise(self, norm_array, original_min, original_max):
        array = (norm_array - self.min_val) / (self.max_val - self.min_val)
        array = array * (original_max - original_min) + original_min
        return array

class Saver:
    """Saver is responsible to save features, and the min max values."""
    def __init__(self, feature_save_dir, save_min_max_values_save_dir):
        self.feature_save_dir = feature_save_dir
        self.save_min_max_values_save_dir = save_min_max_values_save_dir

    def save_feature(self, feature, file_path):
        save_path = self._generate_save_path(file_path)
        np.save(save_path, feature)
        return save_path

    def _generate_save_path(self, file_path):
        file_name = os.path.split(file_path)[1]
        save_path = os.path.join(self.feature_save_dir, file_name + ".npy")
        return save_path

    def save_min_max_values(self, min_max_values):
        save_path = os.path.join(self.save_min_max_values_save_dir, "min_max_values.pkl")
        self._save(min_max_values, save_path)

    @staticmethod
    def _save(data, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(data, f)


class PreprocessingPipeline:
    """PreprocessingPipeline processes audio files in a directory,
    applying the following steps to each file:
        1- load a file
        2- pad the signal (if necessary)
        3- extracting log spectrogram from signal
        4- normalize spectrogram
        5- save the normalised spectrogram

    Storing the min max values of all the log spectrograms.
    """

    def __init__(self):
        self._loader = None
        self.padder = None
        self.extractor = None
        self.normaliser = None
        self.saver = None
        self.min_max_values = {}
        self._num_expected_samples = None

    @property
    def loader(self):
        return self._loader


    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self._num_expected_samples = int(loader.sample_rate * loader.duration)


    def process(self, audio_files_dir):
        for root, _, files in os.walk(audio_files_dir):
            for file in files:
                file_path = os.path.join(root, file)
                self._process_file(file_path)
                #print(f"Processed file {file_path}")
        self.saver.save_min_max_values(self.min_max_values)


    def _process_file(self, file_path):
        signal = self.loader.load(file_path)
        if self._is_padding_necessary(signal):
            signal = self._apply_padding(signal)
        feature = self.extractor.extract(signal)
        norm_feature = self.normaliser.normalise(feature)
        if not np.isnan(norm_feature).any():
            save_path = self.saver.save_feature(norm_feature, file_path)
            self._store_min_max_value(save_path, feature.min(), feature.max())
        else: 
            print("!!!!!!!!!!!!!!! Data is corrupted. Count of nan values: ", np.count_nonzero(np.isnan(norm_feature)))


    def _is_padding_necessary(self, signal):
        if len(signal) < self._num_expected_samples:
            return True
        return False


    def _apply_padding(self, signal):
        num_missing_samples = self._num_expected_samples - len(signal)
        padded_signal = self.padder.right_pad(signal, num_missing_samples)
        return padded_signal


    def _store_min_max_value(self, save_path, min_val, max_val):
        self.min_max_values[save_path] = {
            "min": min_val,
            "max": max_val
        }


def load_fsdd(spectrograms_path):
    x_train = []
    file_paths = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path) # (n_bins, n_frames, 1)
            x_train.append(spectrogram)
            file_paths.append(file_path)
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis] # -> (3000, 256, 64, 1)
    return x_train, file_paths


if __name__ == "__main__":
    FRAME_SIZE = 512
    HOP_LENGTH = 256
    DURATION = 0.74 # in seconds
    SAMPLE_RATE = 22050
    MONO = True

    SPECTOGRAMS_SAVE_DIR = os.path.join("dataset", "fsdd", "spectrograms") 
    MIN_MAX_VALUES_SAVE_DIR = os.path.join("dataset", "fsdd")
    FILES_DIR = os.path.join("dataset", "fsdd", "recordings")

    # instantiate all objects
    loader = Loader(SAMPLE_RATE, DURATION, MONO)
    padder = Padder()
    log_spectogram_extractor = LogSpectogramExtractor(FRAME_SIZE, HOP_LENGTH)
    min_max_normaliser = MinMaxNormaliser(0, 1)
    saver = Saver(SPECTOGRAMS_SAVE_DIR, MIN_MAX_VALUES_SAVE_DIR)

    preprocessing_pipeline = PreprocessingPipeline()
    preprocessing_pipeline.loader = loader
    preprocessing_pipeline.padder = padder
    preprocessing_pipeline.extractor = log_spectogram_extractor
    preprocessing_pipeline.normaliser = min_max_normaliser
    preprocessing_pipeline.saver = saver

    preprocessing_pipeline.process(FILES_DIR)


    # Convert the preprocessed spectrograms to a lighter file format
    SPECTOGRAMS_PATH = os.path.join("dataset", "fsdd", "spectrograms")
    x_train, filepaths = load_fsdd(SPECTOGRAMS_PATH)
    with open(os.path.join('dataset', 'fsdd', 'pickled_spectrograms.npy'), 'wb') as f:
        np.save(f, x_train)

    with open(os.path.join('dataset', 'fsdd', 'pickled_filepaths.npy'), 'wb') as f:
        np.save(f, filepaths)
