import librosa
import matplotlib.pyplot as plt
import math, pickle, os
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from utils.display import *
from utils.audio import AudioProcessor
from utils.generic_utils import load_config
from tqdm import tqdm
from models.wavernn import Model


torch.backends.cudnn.benchmark = True

# define data classes
class MyDataset(Dataset):
    def __init__(self, ids, path):
        self.path = path
        self.metadata = ids

    def __getitem__(self, index):
        file = self.metadata[index]
        m = np.load(f"{self.path}mel/{file}.npy")
        x = np.load(f"{self.path}quant/{file}.npy")
        assert x.max() < 2 ** bits, "{} -- value: {}".format(
            f"{self.path}quant/{file}.npy", x.max()
        )
        assert x.min() >= 0, "{} -- value: {}".format(
            f"{self.path}quant/{file}.npy", x.min()
        )
        return m, x

    def __len__(self):
        return len(self.metadata)


def collate(batch):
    pad = 2
    mel_win = seq_len // ap.hop_length + 2 * pad
    max_offsets = [x[0].shape[-1] - (mel_win + 2 * pad) for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [(offset + pad) * ap.hop_length for offset in mel_offsets]
    mels = [
        x[0][:, mel_offsets[i] : mel_offsets[i] + mel_win] for i, x in enumerate(batch)
    ]
    coarse = [
        x[1][sig_offsets[i] : sig_offsets[i] + seq_len + 1] for i, x in enumerate(batch)
    ]
    mels = np.stack(mels).astype(np.float32)
    coarse = np.stack(coarse).astype(np.int64)
    mels = torch.FloatTensor(mels)
    coarse = torch.LongTensor(coarse)
    # normalize the netowkr input - audio signal
    x_input = 2 * coarse[:, :seq_len].float() / (2 ** bits - 1.) - 1.
    y_coarse = coarse[:, 1:]
    return x_input, mels, y_coarse


def train(
    model, optimizer, criterion, epochs, batch_size, classes, seq_len, step, lr=1e-4
):
    loss_threshold = 4.0
    trn_loader = DataLoader(
        dataset,
        collate_fn=collate,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
    )
    for p in optimizer.param_groups:
        p["lr"] = lr
    for e in range(epochs):
        running_loss = 0.
        val_loss = 0.
        start = time.time()
        running_loss = 0.
        iters = len(trn_loader)
        for i, (x, m, y) in enumerate(trn_loader):
            x, m, y = x.cuda(), m.cuda(), y.cuda()
            y_hat = model(x, m)
            y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
            y = y.unsqueeze(-1)
            loss = criterion(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            speed = (i + 1) / (time.time() - start)
            avg_loss = running_loss / (i + 1)
            step += 1
            if step % 10 == 0:
                print(
                    "Epoch: {}/{} -- Batch: {}/{} -- Loss: {:.3f}"
                    " -- Speed: {:.2f} steps/sec -- Step: {}".format(
                        e + 1, epochs, i + 1, iters, avg_loss, speed, step
                    )
                )
        torch.save(model.state_dict(), MODEL_PATH)
        np.save(STEP_PATH, step)
        print(" <saved>")


def generate(samples=3):
    global output
    k = step // 1000
    test_mels = [np.load(f"{DATA_PATH}mel/{id}.npy") for id in test_ids[:samples]]
    ground_truth = [np.load(f"{DATA_PATH}quant/{id}.npy") for id in test_ids[:samples]]
    for i, (gt, mel) in enumerate(zip(ground_truth, test_mels)):
        print("\nGenerating: %i/%i" % (i + 1, samples))
        gt = 2 * gt.astype(np.float32) / (2 ** bits - 1.) - 1.
        librosa.output.write_wav(
            f"{GEN_PATH}{k}k_steps_{i}_target.wav", gt, sr=ap.sample_rate
        )
        output = model.generate(mel)
        output = ap.mu_decoder(output, 2**bits)
        librosa.output.write_wav(f"{GEN_PATH}{k}k_steps_{i}_generated.wav", output, ap.sample_rate)


if __name__ == "__main__":
    config_path = "../TTS/config.json"
    CONFIG = load_config(config_path)

    ap = AudioProcessor(
        CONFIG.sample_rate,
        CONFIG.num_mels,
        CONFIG.min_level_db,
        CONFIG.frame_shift_ms,
        CONFIG.frame_length_ms,
        CONFIG.ref_level_db,
        CONFIG.num_freq,
        CONFIG.power,
        CONFIG.preemphasis,
        griffin_lim_iters=50,
    )

    bits = 9
    seq_len = ap.hop_length * 5
    run_name = "wavernn_ljspeech"

    # set paths
    MODEL_PATH = f"model_checkpoints/{run_name}.pyt"
    DATA_PATH = f"data/{run_name}/"
    STEP_PATH = f"model_checkpoints/{run_name}_step.npy"
    GEN_PATH = f"model_outputs/{run_name}/"

    # create paths
    os.makedirs("model_checkpoints/", exist_ok=True)
    os.makedirs(GEN_PATH, exist_ok=True)

    # read meta data
    with open(f"{DATA_PATH}dataset_ids.pkl", "rb") as f:
        dataset_ids = pickle.load(f)

    test_ids = dataset_ids[-50:]
    dataset_ids = dataset_ids[:-50]

    # create data loader
    dataset = MyDataset(dataset_ids, DATA_PATH)
    data_loader = DataLoader(
        dataset, collate_fn=collate, batch_size=32, num_workers=0, shuffle=True
    )

    # create the model
    model = Model(
        rnn_dims=512,
        fc_dims=512,
        bits=bits,
        pad=2,
        upsample_factors=(5, 5, 11),
        feat_dims=80,
        compute_dims=128,
        res_out_dims=128,
        res_blocks=10,
    ).cuda()

    # restore any checkpoint
    if not os.path.exists(MODEL_PATH):
        torch.save(model.state_dict(), MODEL_PATH)
    model.load_state_dict(torch.load(MODEL_PATH))

    global step
    step = 0
    if not os.path.exists(STEP_PATH):
        np.save(STEP_PATH, step)
    step = np.load(STEP_PATH)

    # define train functions
    criterion = nn.NLLLoss().cuda()
    optimizer = optim.Adam(model.parameters())
    model.train()

    # HIT IT!!!
    train(
        model,
        optimizer,
        criterion,
        epochs=1000,
        batch_size=16,
        classes=2 ** bits,
        seq_len=seq_len,
        step=step,
        lr=1e-4,
    )

    # generate example
    generate()
