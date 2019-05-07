import time
import numpy as np
from torch import optim
import torch.nn.functional as F
from utils.display import stream, simple_table
from utils.dataset import get_vocoder_datasets
from utils.distribution import discretized_mix_logistic_loss
import hparams as hp
from models.fatchord_version import WaveRNN
from gen_wavernn import gen_testset
from utils.paths import Paths
import argparse


def voc_train_loop(model, loss_func, optimiser, train_set, test_set, lr, total_steps):

    for p in optimiser.param_groups: p['lr'] = lr

    total_iters = len(train_set)
    epochs = (total_steps - model.get_step()) // total_iters + 1

    for e in range(1, epochs + 1):

        start = time.time()
        running_loss = 0.

        for i, (x, y, m) in enumerate(train_set, 1):
            x, m, y = x.cuda(), m.cuda(), y.cuda()

            y_hat = model(x, m)

            if model.mode == 'RAW' :
                y_hat = y_hat.transpose(1, 2).unsqueeze(-1)

            elif model.mode == 'MOL' :
                y = y.float()

            y = y.unsqueeze(-1)


            loss = loss_func(y_hat, y)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running_loss += loss.item()

            speed = i / (time.time() - start)
            avg_loss = running_loss / i

            step = model.get_step()
            k = step // 1000

            if step % hp.voc_checkpoint_every == 0 :
                gen_testset(model, test_set, hp.voc_gen_at_checkpoint, hp.voc_gen_batched,
                            hp.voc_target, hp.voc_overlap, paths.voc_output)
                model.checkpoint(paths.voc_checkpoints)

            msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Loss: {avg_loss:.4f} | {speed:.1f} steps/s | Step: {k}k | '
            stream(msg)

        model.save(paths.voc_latest_weights)
        model.log(paths.voc_log, msg)
        print(' ')


if __name__ == "__main__" :

    # Parse Arguments
    parser = argparse.ArgumentParser(description='Train WaveRNN Vocoder')
    parser.add_argument('--lr', '-l', type=float,  help='[float] override hparams.py learning rate')
    parser.add_argument('--batch_size', '-b', type=int, help='[int] override hparams.py batch size')
    parser.add_argument('--force_train', '-f', action='store_true', help='Forces the model to train past total steps')
    parser.add_argument('--gta', '-g', action='store_true', help='train wavernn on GTA features')
    parser.set_defaults(lr=hp.voc_lr)
    parser.set_defaults(batch_size=hp.voc_batch_size)
    args = parser.parse_args()

    batch_size = args.batch_size
    force_train = args.force_train
    train_gta = args.gta
    lr = args.lr

    print('\nInitialising Model...\n')

    # Instantiate WaveRNN Model
    voc_model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
                        fc_dims=hp.voc_fc_dims,
                        bits=hp.bits,
                        pad=hp.voc_pad,
                        upsample_factors=hp.voc_upsample_factors,
                        feat_dims=hp.num_mels,
                        compute_dims=hp.voc_compute_dims,
                        res_out_dims=hp.voc_res_out_dims,
                        res_blocks=hp.voc_res_blocks,
                        hop_length=hp.hop_length,
                        sample_rate=hp.sample_rate,
                        mode=hp.voc_mode).cuda()

    # Check to make sure the hop length is correctly factorised
    assert np.cumprod(hp.voc_upsample_factors)[-1] == hp.hop_length

    paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)

    voc_model.restore(paths.voc_latest_weights)

    optimiser = optim.Adam(voc_model.parameters())

    train_set, test_set = get_vocoder_datasets(paths.data, batch_size, train_gta)

    total_steps = 10_000_000 if force_train else hp.voc_total_steps

    simple_table([('Remaining', str((total_steps - voc_model.get_step())//1000) + 'k Steps'),
                  ('Batch Size', batch_size),
                  ('LR', lr),
                  ('Sequence Len', hp.voc_seq_len),
                  ('GTA Train', train_gta)])

    loss_func = F.cross_entropy if voc_model.mode == 'RAW' else discretized_mix_logistic_loss

    voc_train_loop(voc_model, loss_func, optimiser, train_set, test_set, lr, total_steps)

    print('Training Complete.')
    print('To continue training increase voc_total_steps in hparams.py or use --force_train')
