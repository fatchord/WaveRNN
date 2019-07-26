import torch
from torch import optim
import torch.nn.functional as F
from utils.display import *
from utils.dataset import get_tts_datasets
import hparams as hp
from utils.text.symbols import symbols
from utils.paths import Paths
from models.tacotron import Tacotron
import argparse
from utils import data_parallel_workaround
import os
from pathlib import Path
import time
import numpy as np


def np_now(x: torch.Tensor): return x.detach().cpu().numpy()


if __name__ == "__main__":

    # Parse Arguments
    parser = argparse.ArgumentParser(description='Train Tacotron TTS')
    parser.add_argument('--force_train', '-f', action='store_true', help='Forces the model to train past total steps')
    parser.add_argument('--force_gta', '-g', action='store_true', help='Force the model to create GTA features')
    parser.add_argument('--force_cpu', '-c', action='store_true', help='Forces CPU-only training, even when in CUDA capable environment')
    args = parser.parse_args()

    force_train = args.force_train
    force_gta = args.force_gta

    if not args.force_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        for session in hp.tts_schedule:
            _, _, _, batch_size = session
            if batch_size % torch.cuda.device_count() != 0:
                raise ValueError('`batch_size` must be evenly divisible by n_gpus!')
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    print('\nInitialising Tacotron Model...\n')

    # Instantiate Tacotron Model
    model = Tacotron(embed_dims=hp.tts_embed_dims,
                     num_chars=len(symbols),
                     encoder_dims=hp.tts_encoder_dims,
                     decoder_dims=hp.tts_decoder_dims,
                     n_mels=hp.num_mels,
                     fft_bins=hp.num_mels,
                     postnet_dims=hp.tts_postnet_dims,
                     encoder_K=hp.tts_encoder_K,
                     lstm_dims=hp.tts_lstm_dims,
                     postnet_K=hp.tts_postnet_K,
                     num_highways=hp.tts_num_highways,
                     dropout=hp.tts_dropout).to(device=device)

    paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)

    model.restore(paths.tts_latest_weights)

    # model.reset_step()

    # model.set_r(hp.tts_r)

    optimizer = optim.Adam(model.parameters())
    if paths.tts_latest_optim.exists():
        print(f'Loading Optimizer State: "{paths.tts_latest_optim}"\n')
        optimizer.load_state_dict(torch.load(paths.tts_latest_optim))

    current_step = model.get_step()

    if not force_gta:

        for session in hp.tts_schedule:

            r, lr, max_step, batch_size = session

            if current_step < max_step:

                train_set, attn_example = get_tts_datasets(paths.data, batch_size, r)

                model.r = r

                training_steps = max_step - current_step

                simple_table([(f'Steps with r={r}', str(training_steps//1000) + 'k Steps'),
                              ('Batch Size', batch_size),
                              ('Learning Rate', lr),
                              ('Outputs/Step (r)', model.r)])

                tts_train_loop(model, optimizer, train_set, lr, training_steps, attn_example)

        print('Training Complete.')
        print('To continue training increase tts_total_steps in hparams.py or use --force_train\n')


    print('Creating Ground Truth Aligned Dataset...\n')

    train_set, attn_example = get_tts_datasets(paths.data, 8, model.r)
    create_gta_features(model, train_set, paths.gta)

    print('\n\nYou can now train WaveRNN on GTA features - use python train_wavernn.py --gta\n')


def tts_train_loop(model: Tacotron, optimizer, train_set, lr, train_steps, attn_example):
    device = next(model.parameters()).device  # use same device as model parameters

    for p in optimizer.param_groups: p['lr'] = lr

    total_iters = len(train_set)
    epochs = train_steps // total_iters + 1

    for e in range(epochs):

        start = time.time()
        running_loss = 0

        for i, (x, m, ids, _) in enumerate(train_set, 1):

            optimizer.zero_grad()

            x, m = x.to(device), m.to(device)

            # Parallelize model onto GPUS using workaround due to python bug
            if device.type == 'cuda' and torch.cuda.device_count() > 1:
                m1_hat, m2_hat, attention = data_parallel_workaround(model, x, m)
            else:
                m1_hat, m2_hat, attention = model(x, m) 

            m1_loss = F.l1_loss(m1_hat, m)
            m2_loss = F.l1_loss(m2_hat, m)

            loss = m1_loss + m2_loss

            running_loss += loss.item()

            loss.backward()
            if hp.tts_clip_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hp.tts_clip_grad_norm)
                if np.isnan(grad_norm):
                    print('grad_norm was NaN!')

            optimizer.step()

            step = model.get_step()
            k = step // 1000

            speed = i / (time.time() - start)

            avg_loss = running_loss / i

            if step % hp.tts_checkpoint_every == 0:
                model.checkpoint(paths.tts_checkpoints, optimizer)

            if attn_example in ids:
                idx = ids.index(attn_example)
                save_attention(np_now(attention[idx][:, :160]), paths.tts_attention/f'{step}')
                save_spectrogram(np_now(m2_hat[idx]), paths.tts_mel_plot/f'{step}', 600)

            msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Loss: {avg_loss:#.4} | {speed:#.2} steps/s | Step: {k}k | '
            stream(msg)

        # Must save latest optimizer state to ensure that resuming training
        # doesn't produce artifacts
        torch.save(optimizer.state_dict(), paths.tts_latest_optim)
        model.save(paths.tts_latest_weights)
        model.log(paths.tts_log, msg)
        print(' ')


def create_gta_features(model: Tacotron, train_set, save_path: Path):
    device = next(model.parameters()).device  # use same device as model parameters

    iters = len(train_set)

    for i, (x, mels, ids, mel_lens) in enumerate(train_set, 1):

        x, mels = x.to(device), mels.to(device)

        with torch.no_grad(): _, gta, _ = model(x, mels)

        gta = gta.cpu().numpy()

        for j, item_id in enumerate(len(ids)):
            mel = gta[j][:, :mel_lens[j]]
            mel = (mel + 4) / 8
            np.save(save_path/f'{item_id}.npy', mel, allow_pickle=False)

        bar = progbar(i, iters)
        msg = f'{bar} {i}/{iters} Batches '
        stream(msg)


def save_checkpoint(paths: Paths, model: Tacotron, optimizer, name=None):
    """Saves the training session to disk.

    Args:
        paths:  Provides information about the different paths to use.
        model:  A `Tacotron` model to save the parameters and buffers from.
        optimizer:  An optmizer to save the state of (momentum, etc).
        name:  If provided, will name to a checkpoint with the given name. Note
            that regardless of whether this is provided or not, this function
            will always update the files specified in `paths` that give the
            location of the latest weights and optimizer state. Saving
            a named checkpoint happens in addition to this update.
    """    
    def helper(path_dict, is_named):
        s = 'named' if is_named else 'latest'
        num_exist = sum(p.exists() for p in path_dict.values())
        
        if num_exist not in (0,2):
            # Checkpoint broken
            raise FileNotFoundError(
                f'We expected either both or no files in the {s} checkpoint to '
                'exist, but instead we got exactly one!')
        
        if num_exist == 0:
            print('Creating {s} checkpoint...')
            for p in path_dict.values():
                p.parent.mkdir(parents=True)
        else:
            print('Saving to existing {s} checkpoint...')
        
        print(f'Saving {s} weights: {path_dict["w"]}')
        model.save(path_dict['w'])
        print(f'Saving {s} optimizer state: {path_dict["o"]}')
        torch.save(optimizer.state_dict(), path_dict['o'])    
    
    latest_paths = {'w': paths.tts_latest_weights, 'o': paths.tts_latest_optim}
    helper(latest_paths, False)

    if name:
        named_paths ={
            'w': paths.tts_checkpoints/f'{name}_weights.pyt',
            'o': paths.tts_checkpoints/f'{name}_optim.pyt',
        }
        helper(named_paths, True)


def restore_checkpoint(paths: Paths, model: Tacotron, optimizer, name=None, create_if_missing=False):
    """Restores from a training session saved to disk.

    Args:
        paths:  Provides information about the different paths to use.
        model:  A `Tacotron` model to save the parameters and buffers from.
        optimizer:  An optmizer to save the state of (momentum, etc).
        name:  If provided, will restore from a checkpoint with the given name.
            Otherwise, will restore from the latest weights and optimizer state
            as specified in `paths`.
        create_if_missing:  If `True`, will create the checkpoint if it doesn't
            yet exist, as well as update the files specified in `paths` that
            give the location of the current latest weights and optimizer state.
            If `False` and the checkpoint doesn't exist, will raise a 
            `FileNotFoundError`.
    """
    if name:
        path_dict = {
            'w': paths.tts_checkpoints/f'{name}_weights.pyt',
            'o': paths.tts_checkpoints/f'{name}_optim.pyt',
        }
        s = 'named'
    else:
        path_dict = {
            'w': paths.tts_latest_weights,
            'o': paths.tts_latest_optim
        }
        s = 'latest'
    
    num_exist = sum(p.exists() for p in path_dict.values())
    if num_exist == 2:
        # Checkpoint exists
        print(f'Restoring from {s} checkpoint...')
        print(f'Loading {s} weights: {path_dict["w"]}')
        model.load(path_dict['w'])
        print(f'Loading {s} optimizer state: {path_dict["o"]}')
        optimizer.load_state_dict(torch.load())
    elif create_if_missing:
        save_checkpoint(paths, model, optimizer, name)
    else:
        raise FileNotFoundError(f'The {s} checkpoint could not be found!')
