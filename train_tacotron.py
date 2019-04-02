import time
import numpy as np
from torch import optim
import torch.nn.functional as F
from utils.display import stream, simple_table
from utils.dataset import get_vocoder_datasets
import hparams as hp
from utils.text.symbols import symbols
from utils.paths import Paths
import argparse


def tts_train_loop(model, optimizer, batch_size, epochs, lr, step, clip_grad=1.0):

    for p in optimizer.param_groups: p['lr'] = lr

    for e in range(epochs):

        start = time.time()
        running_loss = 0

        sampler = BinnedLength(frame_lengths, batch_size=batch_size, bin_size=bin_size)

        loader = DataLoader(dataset, collate_fn=collate, batch_size=batch_size,
                            sampler=sampler, num_workers=1, pin_memory=True)

        for i, (x, m, _) in enumerate(loader):

            optimizer.zero_grad()

            x = Variable(x).cuda()
            m = Variable(m).cuda()

            m1_hat, m2_hat, attention = model(x, m)

            m1_loss = F.l1_loss(m1_hat, m)
            m2_loss = F.l1_loss(m2_hat, m)

            loss = m1_loss + m2_loss

            running_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

            step += 1

            speed = (i + 1) / (time.time() - start)
            t = time_since(start)
            l = running_loss / (i + 1)


if __name__ == "__main__" :

    # Parse Arguments
    parser = argparse.ArgumentParser(description='Train Tacotron TTS')
    parser.add_argument('--lr', '-l', type=float,  help='[float] override hparams.py learning rate')
    parser.add_argument('--batch_size', '-b', type=int, help='[int] override hparams.py batch size')
    parser.add_argument('--force_train', '-f', action='store_true', help='Forces the model to train past total steps')
    parser.set_defaults(lr=hp.tts_lr)
    parser.set_defaults(batch_size=hp.tts_batch_size)
    args = parser.parse_args()

    batch_size = args.batch_size
    force_train = args.force_train
    lr = args.lr

    print('\nInitialising Tacotron Model...\n')

    # Instantiate Tacotron Model
    model = Tacotron(r=hp.tts_r,
                     embed_dims=hp.tts_embed_dims,
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
                     dropout=hp.tts_dropout).cuda()

    # Check to make sure the hop length is correctly factorised
    assert np.cumprod(hp.voc_upsample_factors)[-1] == hp.hop_length

    paths = Paths(hp.data_path, hp.model_id)

    voc_model.restore(paths.voc_latest_weights)

    optimiser = optim.Adam(voc_model.parameters())

    train_set, test_set = get_vocoder_datasets(paths.data, batch_size)

    total_steps = 10_000_000 if force_train else hp.voc_total_steps

    simple_table([('Steps Remaining', str((total_steps - voc_model.get_step())//1000) + 'k'),
                  ('Batch Size', batch_size),
                  ('Learning Rate', lr),
                  ('Sequence Length', hp.voc_seq_len)])

    voc_train_loop(voc_model, optimiser, train_set, test_set, lr, total_steps)

    print('Training Complete.')
    print('To continue training increase tts_total_steps in hparams.py or use --force_train')
