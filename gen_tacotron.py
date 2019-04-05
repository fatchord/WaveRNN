import torch
from models.fatchord_wavernn import Model
import hparams as hp
from utils.text.symbols import symbols
from utils.paths import Paths
from models.tacotron import Tacotron
import argparse
from utils.text import text_to_sequence
from utils.display import save_attention

if __name__ == "__main__" :

    # Parse Arguments
    parser = argparse.ArgumentParser(description='TTS Generator')
    parser.add_argument('--text', '-t', type=str, help='[string] Type in something here and TTS will generate it!')
    parser.add_argument('--batched', '-b', dest='batched', action='store_true', help='Fast Batched Generation')
    parser.add_argument('--unbatched', '-u', dest='batched', action='store_false', help='Slow Unbatched Generation')
    parser.set_defaults(batched=hp.voc_gen_batched)
    parser.set_defaults(text=None)
    args = parser.parse_args()

    batched = args.batched
    custom_text = args.text

    paths = Paths(hp.data_path, hp.model_id)

    print('\nInitialising WaveRNN Model...\n')

    # Instantiate WaveRNN Model
    voc_model = Model(rnn_dims=hp.voc_rnn_dims,
                      fc_dims=hp.voc_fc_dims,
                      bits=hp.bits,
                      pad=hp.voc_pad,
                      upsample_factors=hp.voc_upsample_factors,
                      feat_dims=hp.num_mels,
                      compute_dims=hp.voc_compute_dims,
                      res_out_dims=hp.voc_res_out_dims,
                      res_blocks=hp.voc_res_blocks,
                      hop_length=hp.hop_length,
                      sample_rate=hp.sample_rate).cuda()

    voc_model.restore(paths.voc_latest_weights)

    print('\nInitialising Tacotron Model...\n')

    # Instantiate Tacotron Model
    tts_model = Tacotron(r=hp.tts_r,
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

    tts_model.restore(paths.tts_latest_weights)

    if custom_text :
        inputs = [text_to_sequence(custom_text.strip(), hp.tts_cleaner_names)]
    else :
        with open('sentences.txt') as f :
            inputs = [text_to_sequence(l.strip(), hp.tts_cleaner_names) for l in f]

    mels = []
    for i, x in enumerate(inputs, 1) :
        print(f'\n| Generating {i}/{len(inputs)}')
        _, m, attention = tts_model.generate(x)
        save_path = f'{paths.tts_output}{i}_bathed_{str(batched)}.wav'
        save_attention(attention, save_path, tts_model.get_step())
        m = torch.tensor(m).unsqueeze(0)
        m = (m + 4) / 8
        voc_model.generate(m, save_path, batched, hp.voc_target, hp.voc_overlap, hp.mu_law)



    print('\n\nDone.\n')