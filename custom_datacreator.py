import librosa
from datasets import Dataset
def custom_dataset(examples):
    def speech_file_to_array_fn(batch):
        speech_array, sampling_rate = librosa.load(batch["audio_filepath"],sr=16000)
        batch["audio"] = speech_array
        batch["sampling_rate"] = sampling_rate
        return batch

    dataset = Dataset.from_dict(examples)
    dataset = dataset.map(speech_file_to_array_fn, remove_columns=["audio_filepath"])
    return dataset
