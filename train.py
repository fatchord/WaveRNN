import time
from torch import optim
import torch.nn.functional as F
from utils.display import stream
from utils.dataset import get_datasets
from utils.dsp import *
import hparams as hp
from models.fatchord_wavernn import Model
from utils.paths import Paths


def train_loop(model, optimiser, train_set, test_set, lr):

    for p in optimiser.param_groups: p['lr'] = lr

    total_iters = len(train_set)
    epochs = (hp.total_steps - model.get_step()) // total_iters + 1

    for e in range(1, epochs + 1):

        start = time.time()
        running_loss = 0.

        for i, (x, y, m) in enumerate(train_set, 1):
            x, m, y = x.cuda(), m.cuda(), y.cuda()

            y_hat = model(x, m)
            y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
            y = y.unsqueeze(-1)
            loss = F.cross_entropy(y_hat, y)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running_loss += loss.item()

            speed = i / (time.time() - start)
            avg_loss = running_loss / i

            step = model.get_step()
            k = step // 1000

            if step % hp.checkpoint_every == 0 :
                generate(test_set, step)
                model.checkpoint(paths.checkpoints)

            msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Loss: {avg_loss:#.3} | {speed:#.2} steps/s | Step: {k}k | '
            stream(msg)

        model.save(paths.latest_weights)
        model.log(paths.log, msg)
        print(' ')


def generate(test_set, step, samples=5, batched=True, target=11_000, overlap=550):

    k = step // 1000

    for i, (m, x) in enumerate(test_set, 1):

        if i > samples : break

        print('\nGenerating: %i/%i' % (i, samples))

        x = label_2_float(x[0].numpy(), hp.bits)
        librosa.output.write_wav(f'{paths.output}{k}k_steps_{i}_target.wav', x, sr=hp.sample_rate)

        batch_str = f'gen_batched_target{target}_overlap{overlap}' if batched else 'gen_NOT_BATCHED'
        save_str = f'{paths.output}{k}k_steps_{i}_{batch_str}.wav'

        _ = model.generate(m, save_str, batched, target, overlap)


print('\nInitialising Model...\n')

model = Model(rnn_dims=hp.rnn_dims,
              fc_dims=hp.fc_dims,
              bits=hp.bits,
              pad=hp.pad,
              upsample_factors=hp.upsample_factors,
              feat_dims=hp.feat_dims,
              compute_dims=hp.compute_dims,
              res_out_dims=hp.res_out_dims,
              res_blocks=hp.res_blocks,
              hop_length=hp.hop_length,
              sample_rate=hp.sample_rate).cuda()

paths = Paths(hp.data_path, hp.model_id)

model.restore(paths.latest_weights)

optimiser = optim.Adam(model.parameters())

train_set, test_set = get_datasets(paths.data, hp.batch_size)

train_loop(model, optimiser, train_set, test_set, hp.lr)
