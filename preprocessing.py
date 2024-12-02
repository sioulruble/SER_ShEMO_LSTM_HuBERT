import os
import torchaudio
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class GeneralDataset(Dataset):
    def __init__(self, male_directory, female_directory, target_sample_rate, max_len=None):
        self.male_files = [(os.path.join(male_directory, f), 'male') for f in os.listdir(male_directory)]
        self.female_files = [(os.path.join(female_directory, f), 'female') for f in os.listdir(female_directory)]
        self.files = self.male_files + self.female_files
        self.target_sample_rate = target_sample_rate
        self.max_len = max_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path, gender = self.files[idx]
        waveform, sample_rate = torchaudio.load(file_path)
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
        waveform = resampler(waveform)
        label = self.extract_label_char_from_filename(file_path)
        return waveform, label

    @staticmethod
    def extract_label_char_from_filename(filename):
        return filename.split('/')[-1][3]

    @staticmethod
    def label_char_to_one_hot(char, label_map):
        return torch.Tensor(label_map[char])

class NADataset(GeneralDataset):
    """
    Subdatasat to only process the 2 main classes from the ShEMO dataset : Angry & Neutral
    """
    def __init__(self, male_directory, female_directory, target_sample_rate, max_len=80000):
        super().__init__(male_directory, female_directory, target_sample_rate, max_len)
        self.files = [f for f in self.files if self.extract_label_char_from_filename(f[0]) in ['N', 'A']]

    @staticmethod
    def label_char_to_one_hot(char):
        label_map = {'N': [0, 1], 'A': [1, 0]}
        return torch.Tensor(label_map[char])

def custom_collate_fn(batch):
    waveforms = [torch.tensor(item[0].T) for item in batch]
    labels = [item[1] for item in batch]
    padded_waveforms = pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    padded_waveforms = padded_waveforms.squeeze(2)
    attention_masks = torch.tensor(
        [[1] * waveform.size(0) + [0] * (padded_waveforms.size(1) - waveform.size(0))
         for waveform in waveforms]
    )
    labels = torch.stack(labels)
    return padded_waveforms, attention_masks, labels
