import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from utils.dsp import *
import hparams as hp

def get_datasets(path, batch_size=16) :

    with open(f'{path}dataset_ids.pkl', 'rb') as f:
        dataset_ids = pickle.load(f)

    test_ids = dataset_ids[-hp.voc_test_samples:]
    train_ids = dataset_ids[:-hp.voc_test_samples]

    train_dataset = AudiobookDataset(train_ids, path)
    test_dataset = AudiobookDataset(test_ids, path)

    train_set = DataLoader(train_dataset,
                           collate_fn=collate,
                           batch_size=batch_size,
                           num_workers=2,
                           shuffle=True,
                           pin_memory=True)

    test_set = DataLoader(test_dataset,
                          batch_size=1,
                          num_workers=1,
                          shuffle=False,
                          pin_memory=True)

    return train_set, test_set


class AudiobookDataset(Dataset):
    def __init__(self, ids, path):
        self.path = path
        self.metadata = ids

    def __getitem__(self, index):
        file = self.metadata[index]
        m = np.load(f'{self.path}mel/{file}.npy')
        x = np.load(f'{self.path}quant/{file}.npy')
        return m, x

    def __len__(self):
        return len(self.metadata)



def collate(batch):
    mel_win = hp.voc_seq_len // hp.hop_length + 2 * hp.voc_pad
    max_offsets = [x[0].shape[-1] - (mel_win + 2 * hp.voc_pad) for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [(offset + hp.voc_pad) * hp.hop_length for offset in mel_offsets]

    mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win] for i, x in enumerate(batch)]

    labels = [x[1][sig_offsets[i]:sig_offsets[i] + hp.voc_seq_len + 1] for i, x in enumerate(batch)]

    mels = np.stack(mels).astype(np.float32)
    labels = np.stack(labels).astype(np.int64)

    mels = torch.FloatTensor(mels)
    labels = torch.LongTensor(labels)

    x = label_2_float(labels[:, :hp.voc_seq_len].float(), hp.bits)

    y = labels[:, 1:]

    return x, y, mels
