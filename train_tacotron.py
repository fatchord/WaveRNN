import torch
from torch import optim
import torch.nn.functional as F
from utils.display import *
from utils.dataset import get_tts_dataset
import hparams as hp
from utils.text.symbols import symbols
from utils.paths import Paths
from models.tacotron import Tacotron
import argparse


def np_now(x): return x.detach().cpu().numpy()


def tts_train_loop(model, optimizer, train_set, lr, train_steps, attn_example):
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

            m1_hat, m2_hat, attention = model(x, m)

            m1_loss = F.l1_loss(m1_hat, m)
            m2_loss = F.l1_loss(m2_hat, m)

            loss = m1_loss + m2_loss

            running_loss += loss.item()

            loss.backward()

            if hp.tts_clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), hp.tts_clip_grad_norm)

            optimizer.step()

            step = model.get_step()
            k = step // 1000

            speed = i / (time.time() - start)

            avg_loss = running_loss / i

            if step % hp.tts_checkpoint_every == 0:
                model.checkpoint(paths.tts_checkpoints)

            if attn_example in ids:
                idx = ids.index(attn_example)
                save_attention(attention[idx][:, :160], f'{paths.tts_attention}{step}')
                save_spectrogram(np_now(m2_hat[idx]), f'{paths.tts_mel_plot}{step}', 600)

            msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Loss: {avg_loss:#.4} | {speed:#.2} steps/s | Step: {k}k | '
            stream(msg)

        model.save(paths.tts_latest_weights)
        model.log(paths.tts_log, msg)
        print(' ')


def create_gta_features(model, train_set, save_path):
    device = next(model.parameters()).device  # use same device as model parameters

    iters = len(train_set)

    for i, (x, mels, ids, mel_lens) in enumerate(train_set, 1):

        x, mels = x.to(device), mels.to(device)

        with torch.no_grad(): _, gta, _ = model(x, mels)

        gta = gta.cpu().numpy()

        for j in range(len(ids)):
            mel = gta[j][:, :mel_lens[j]]
            mel = (mel + 4) / 8
            id = ids[j]
            np.save(f'{save_path}{id}.npy', mel, allow_pickle=False)

        bar = progbar(i, iters)
        msg = f'{bar} {i}/{iters} Batches '
        stream(msg)


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

    optimiser = optim.Adam(model.parameters())

    current_step = model.get_step()

    if not force_gta:

        for session in hp.tts_schedule:

            r, lr, max_step, batch_size = session

            if current_step < max_step:

                train_set, attn_example = get_tts_dataset(paths.data, batch_size, r)

                model.set_r(r)

                training_steps = max_step - current_step

                simple_table([(f'Steps with r={r}', str(training_steps//1000) + 'k Steps'),
                              ('Batch Size', batch_size),
                              ('Learning Rate', lr),
                              ('Outputs/Step (r)', model.get_r())])

                tts_train_loop(model, optimiser, train_set, lr, training_steps, attn_example)

        print('Training Complete.')
        print('To continue training increase tts_total_steps in hparams.py or use --force_train\n')


    print('Creating Ground Truth Aligned Dataset...\n')

    train_set, attn_example = get_tts_dataset(paths.data, 8, model.get_r())
    create_gta_features(model, train_set, paths.gta)

    print('\n\nYou can now train WaveRNN on GTA features - use python train_wavernn.py --gta\n')
