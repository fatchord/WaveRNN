import argparse
import math
import os
import pickle
import shutil
import sys
import traceback

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset import MyDataset
from distribute import *
from models.losses import gaussian_loss, sample_from_gaussian
from models.wavernn import Model
from utils.audio import AudioProcessor
from utils.display import *
from utils.generic_utils import (count_parameters, load_config,
                                 remove_experiment_folder, save_checkpoint,
                                 check_update)

sys.path.insert(0, "/home/erogol/projects/")



torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(54321)
use_cuda = torch.cuda.is_available()
num_gpus = torch.cuda.device_count()
print(" > Using CUDA: ", use_cuda, flush=True)
print(" > Number of GPUs: ", num_gpus, flush=True)


def setup_loader(is_val=False):
    global train_ids
    dataset = MyDataset(
        test_ids if is_val else train_ids,
        DATA_PATH,
        CONFIG.mel_len,
        ap.hop_length,
        ap.bits,
        CONFIG.pad,
        ap,
        is_val,
    )
    sampler = DistributedSampler(dataset) if num_gpus > 1 else None
    loader = DataLoader(
        dataset,
        collate_fn=dataset.collate,
        batch_size=CONFIG.batch_size,
        num_workers=0,
        # shuffle=True,
        pin_memory=True,
        sampler=sampler,
    )
    return loader


def train(model, optimizer, criterion, scheduler, epochs, batch_size, classes, step, lr, args):
    global CONFIG
    global train_ids
    # create train loader
    train_loader = setup_loader(False)

    for p in optimizer.param_groups:
        p["initial_lr"] = lr
        p["lr"] = lr

    for e in range(epochs):
        running_loss = 0.0
        # TODO: write validation iteration
        # val_loss = 0.
        start = time.time()
        running_loss = 0.0
        iters = len(train_loader)
        # train loop
        print(" > Training", flush=True)
        model.train()
        for i, (x, m, y) in enumerate(train_loader):
            if use_cuda:
                x, m, y = x.cuda(), m.cuda(), y.cuda()
            scheduler.step()
            optimizer.zero_grad()
            y_hat = model(x, m)
            # y_hat = y_hat.transpose(1, 2)
            y = y.unsqueeze(-1)
            m_scaled, _ = model.upsample(m)
            loss = criterion(y_hat, y)
            grad_norm, skip_flag = check_update(model, 2) 
            if not skip_flag:           
                loss.backward()
                optimizer.step()
            speed = (i + 1) / (time.time() - start)
            step += 1
            cur_lr = optimizer.param_groups[0]["lr"]
            # Compute avg loss
            if num_gpus > 1:
                loss = reduce_tensor(loss.data, num_gpus)
            running_loss += loss.item()
            avg_loss = running_loss / (i + 1)
            if step % CONFIG.print_step == 0:
                print(
                    " | > Epoch: {}/{} -- Batch: {}/{} -- Loss: {:.3f}"
                    " -- Speed: {:.2f} steps/sec -- Step: {} -- lr: {} -- GradNorm: {}".format(
                        e + 1, epochs, i + 1, iters, avg_loss, speed, step, cur_lr, grad_norm
                    )
                )
            if step % CONFIG.checkpoint_step == 0 and args.rank == 0:
                save_checkpoint(model, optimizer, avg_loss, MODEL_PATH, step, e)
                print(" > checkpoint saved", flush=True)
        # visual
        # m_scaled, _ = model.upsample(m)
        # plot_spec(m[0], VIS_PATH + "/mel_{}.png".format(step))
        # plot_spec(
        #     m_scaled[0].transpose(0, 1), VIS_PATH + "/mel_scaled_{}.png".format(step)
        # )
        # validation loop
        evaluate(model, criterion, batch_size)


def evaluate(model, criterion, batch_size):
    global CONFIG
    global test_ids
    # create train loader
    val_loader = setup_loader(True)

    running_val_loss = 0.0
    iters = len(val_loader)
    # train loop
    print(" > Validation", flush=True)
    model.eval()
    val_step = 0
    with torch.no_grad():
        for i, (x, m, y) in enumerate(val_loader):
            if use_cuda:
                x, m, y = x.cuda(), m.cuda(), y.cuda()
            y_hat = model(x, m)
            # y_hat = y_hat.transpose(1, 2)
            y = y.unsqueeze(-1)
            loss = criterion(y_hat, y)
            # Compute avg loss
            if num_gpus > 1:
                loss = reduce_tensor(loss.data, num_gpus)
            running_val_loss += loss.item()
            avg_val_loss = running_val_loss / (i + 1)
            val_step += 1
            if val_step % CONFIG.print_step == 0:
                print(
                    " | > Batch: {}/{} -- Loss: {:.3f}".format(
                        iters, val_step, avg_val_loss
                    )
                )
        print(" | > Validation Loss: {}".format(avg_val_loss), flush=True)


def generate(step, samples=1, mulaw=False):
    global output
    k = step // 1000
    test_mels = [np.load(f"{DATA_PATH}mel/{test_id}.npy")]
    ground_truth = [np.load(f"{DATA_PATH}quant/{test_id}.npy")]
    for i, (gt, mel) in enumerate(zip(ground_truth, test_mels)):
        print("\nGenerating: %i/%i" % (i + 1, samples))
        gt = 2 * gt.astype(np.float32) / (2 ** bits - 1.0) - 1.0
        librosa.output.write_wav(
            f"{GEN_PATH}{k}k_steps_{i}_target.wav", gt, sr=ap.sample_rate
        )
        output = model.generate(mel)
        if mulaw:
            output = ap.mulaw_decoder(output, 2 ** bits)
        librosa.output.write_wav(
            f"{GEN_PATH}{k}k_steps_{i}_generated.wav", output, ap.sample_rate
        )


def main(args):
    global train_ids
    global test_ids

    # read meta data
    with open(f"{DATA_PATH}/dataset_ids.pkl", "rb") as f:
        train_ids = pickle.load(f)

    # pick validation set
    test_ids = train_ids[-10:]
    test_id = train_ids[4]
    train_ids = train_ids[:-10]

    # create the model
    model = Model(
        rnn_dims=512,
        fc_dims=512,
        bits=ap.bits,
        pad=CONFIG.pad,
        upsample_factors=(5, 5, 11),
        feat_dims=80,
        compute_dims=128,
        res_out_dims=128,
        res_blocks=10,
        hop_length=ap.hop_length,
        sample_rate=ap.sample_rate,
    ).cuda()

    num_parameters = count_parameters(model)
    print(" > Number of model parameters: {}".format(num_parameters), flush=True)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG.lr)
    
    # slow start for the first 5 epochs
    lr_lambda = lambda epoch: min(epoch / CONFIG.warmup_steps , 1)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    step = 0
    # restore any checkpoint
    if args.restore_path:
        checkpoint = torch.load(args.restore_path)
        try:
            model.load_state_dict(checkpoint["model"])
            # TODO: fix resetting restored optimizer lr 
            # optimizer.load_state_dict(checkpoint["optimizer"])
        except:
            model_dict = model.state_dict()
            # Partial initialization: if there is a mismatch with new and old layer, it is skipped.
            # 1. filter out unnecessary keys
            pretrained_dict = {
                k: v for k, v in checkpoint["model"].items() if k in model_dict
            }
            # 2. filter out different size layers
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if v.numel() == model_dict[k].numel()
            }
            # 3. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 4. load the new state dict
            model.load_state_dict(model_dict)
            print(
                " | > {} / {} layers are initialized".format(
                    len(pretrained_dict), len(model_dict)
                )
            )
        step = checkpoint["step"]

    # DISTRIBUTED
    if num_gpus > 1:
        model = apply_gradient_allreduce(model)

    # define train functions
    criterion = gaussian_loss
    model.train()

    # HIT IT!!!
    train(
        model,
        optimizer,
        criterion,
        scheduler,
        epochs=CONFIG.epochs,
        batch_size=CONFIG.batch_size,
        classes=2 ** bits,
        step=step,
        lr=CONFIG.lr * num_gpus,
        args=args,
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
        "--data_path", type=str, default="", help="data path to overwrite config.json."
    )
    parser.add_argument(
        "--output_path", type=str, help="path for training outputs.", default=""
    )
    # DISTRUBUTED
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="DISTRIBUTED: process rank for distributed training.",
    )
    parser.add_argument(
        "--group_id", type=str, default="", help="DISTRIBUTED: process group id."
    )

    args = parser.parse_args()
    CONFIG = load_config(args.config_path)

    if args.data_path != "":
        CONFIG.data_path = args.data_path
    DATA_PATH = CONFIG.data_path

    # DISTRUBUTED
    if num_gpus > 1:
        init_distributed(
            args.rank,
            num_gpus,
            args.group_id,
            CONFIG.distributed["backend"],
            CONFIG.distributed["url"],
        )

    global ap
    ap = AudioProcessor(**CONFIG.audio)
    bits = CONFIG.audio["bits"]

    # setup output paths and read configs
    _ = os.path.dirname(os.path.realpath(__file__))
    if args.data_path != "":
        CONFIG.data_path = args.data_path

    if args.output_path == "":
        OUT_PATH = os.path.join(_, CONFIG.output_path)
    else:
        OUT_PATH = args.output_path

    if args.group_id == "":
        OUT_PATH = create_experiment_folder(OUT_PATH, CONFIG.model_name)

    AUDIO_PATH = os.path.join(OUT_PATH, "test_audios")

    if args.rank == 0:
        # set paths
        MODEL_PATH = f"{OUT_PATH}/model_checkpoints/"
        GEN_PATH = f"{OUT_PATH}/model_outputs/"
        VIS_PATH = f"{OUT_PATH}/visual/"
        shutil.copyfile(args.config_path, os.path.join(OUT_PATH, "config.json"))

        # create paths
        os.makedirs(MODEL_PATH, exist_ok=True)
        os.makedirs(GEN_PATH, exist_ok=True)
        os.makedirs(VIS_PATH, exist_ok=True)

        os.makedirs(AUDIO_PATH, exist_ok=True)
        shutil.copyfile(args.config_path, os.path.join(OUT_PATH, "config.json"))
        os.chmod(AUDIO_PATH, 0o775)
        os.chmod(OUT_PATH, 0o775)

    try:
        main(args)
    except KeyboardInterrupt:
        remove_experiment_folder(OUT_PATH)
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    except Exception:
        remove_experiment_folder(OUT_PATH)
        traceback.print_exc()
        sys.exit(1)
