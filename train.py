import librosa
import shutil
import argparse
import matplotlib.pyplot as plt
import math, pickle, os
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from utils.display import *
try:
    from TTS.utils.audio import AudioProcessor
except:
    from utils.audio import AudioProcessor    
from utils.generic_utils import load_config, save_checkpoint, AnnealLR
from tqdm import tqdm
from models.wavernn import Model

use_cuda = torch.cuda.is_available()
if use_cuda:
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
        x[1][sig_offsets[i] - 1 : sig_offsets[i] + seq_len] for i, x in enumerate(batch)
    ]
    mels = np.stack(mels).astype(np.float32)
    coarse = np.stack(coarse).astype(np.int64)
    mels = torch.FloatTensor(mels)
    coarse = torch.LongTensor(coarse)
    x_input = 2 * coarse[:, :seq_len].float() / (2 ** bits - 1.) - 1.
    y_coarse = coarse[:, 1:]
    return x_input, mels, y_coarse


def train(model, optimizer, criterion, epochs, batch_size, classes, seq_len, step, lr):
    global CONFIG
    # loss_threshold = 4.0
    # create train loader
    dataset = MyDataset(dataset_ids, DATA_PATH)
    train_loader = DataLoader(
        dataset,
        collate_fn=collate,
        batch_size=batch_size,
        num_workers=CONFIG.num_workers,
        shuffle=True,
        pin_memory=True,
    )

    for p in optimizer.param_groups:
        p["initial_lr"] = lr

    scheduler = AnnealLR(optimizer, warmup_steps=CONFIG.warmup_steps, last_epoch=step)
    for e in range(epochs):
        running_loss = 0.
        # TODO: write validation iteration
        # val_loss = 0.
        start = time.time()
        running_loss = 0.
        iters = len(train_loader)
        # train loop
        print(" > Training")
        model.train()
        for i, (x, m, y) in enumerate(train_loader):
            if use_cuda:
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
            scheduler.step()
            cur_lr = optimizer.param_groups[0]["lr"]
            if step % CONFIG.print_step == 0:
                print(
                    " | > Epoch: {}/{} -- Batch: {}/{} -- Loss: {:.3f}"
                    " -- Speed: {:.2f} steps/sec -- Step: {} -- lr: {:}".format(
                        e + 1, epochs, i + 1, iters, avg_loss, speed, step, cur_lr
                    )
                )
            if step % CONFIG.checkpoint_step == 0:
                save_checkpoint(model, optimizer, avg_loss, MODEL_PATH, step, e)
                print(" > modelsaved")
        # validation loop
        running_val_loss = 0
        generate(step)


def generate(step, samples=1, mulaw=False):
    global output
    k = step // 1000
    test_mels = [np.load(f"{DATA_PATH}mel/{test_id}.npy")]
    ground_truth = [np.load(f"{DATA_PATH}quant/{test_id}.npy")]
    # test_mels = [np.load(f"{DATA_PATH}mel/{id}.npy") for id in test_ids[:samples]]
    # ground_truth = [np.load(f"{DATA_PATH}quant/{id}.npy") for id in test_ids[:samples]]
    for i, (gt, mel) in enumerate(zip(ground_truth, test_mels)):
        print("\nGenerating: %i/%i" % (i + 1, samples))
        gt = 2 * gt.astype(np.float32) / (2 ** bits - 1.) - 1.
        librosa.output.write_wav(
            f"{GEN_PATH}{k}k_steps_{i}_target.wav", gt, sr=ap.sample_rate
        )
        output = model.module.generate(mel)
        if mulaw:
            output = ap.mulaw_decoder(output, 2 ** bits)
        librosa.output.write_wav(
            f"{GEN_PATH}{k}k_steps_{i}_generated.wav", output, ap.sample_rate
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, help="path to config file for training."
    )
    parser.add_argument(
        "--restore_path", type=str, default=0, help="path for a model to fine-tune."
    )
    parser.add_argument(
        "--data_path", type=str, default='', help="data path to overwrite config.json."
    )
    
    args = parser.parse_args()
    CONFIG = load_config(args.config_path)

    if args.data_path != '':
        CONFIG.data_path = args.data_path

    ap = AudioProcessor(**CONFIG.audio)

    bits = CONFIG.audio['bits']
    seq_len = ap.hop_length * 5
    run_name = CONFIG.run_name

    # set paths
    OUT_PATH = os.path.join(CONFIG.out_path, CONFIG.run_name)
    MODEL_PATH = f"{OUT_PATH}/model_checkpoints/"
    DATA_PATH = f"{OUT_PATH}/data/"
    GEN_PATH = f"{OUT_PATH}/model_outputs/"
    shutil.copyfile(args.config_path, os.path.join(OUT_PATH, "config.json"))

    # create paths
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(GEN_PATH, exist_ok=True)

    # read meta data
    with open(f"{DATA_PATH}dataset_ids.pkl", "rb") as f:
        dataset_ids = pickle.load(f)

    test_ids = dataset_ids[-50:]
    test_id = test_ids[1]
    dataset_ids = dataset_ids[:-50]

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
    )
    if use_cuda:
        model = nn.DataParallel(model).cuda()
    optimizer = optim.Adam(model.parameters())

    step = 0
    # restore any checkpoint
    if args.restore_path:
        checkpoint = torch.load(args.restore_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = checkpoint["step"]

    # define train functions
    criterion = nn.NLLLoss().cuda()
    model.train()

    # HIT IT!!!
    train(
        model,
        optimizer,
        criterion,
        epochs=1000,
        batch_size=CONFIG.batch_size,
        classes=2 ** bits,
        seq_len=seq_len,
        step=step,
        lr=CONFIG.lr,
    )
