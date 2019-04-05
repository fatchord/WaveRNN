
# CONFIG -----------------------------------------------------------------------------------------------------------#

# Here are the input and output data paths (Note: you can override wav_path in preprocess.py)
wav_path = '/path/to/wav_files/'
data_path = 'data/'

# model ids are split - that way you can train a new tts with an old wavernn and vice versa
voc_model_id = 'ljspeech_9bit_mulaw'
tts_model_id = 'ljspeech_9bit_mulaw'

# use this if you want a quick start i.e., use pretrained models with gen_tacotron.py
# allow_pretrained = True
# voc_pretrained = 'pretrained/ljspeech_700k_gta.tar.gz'
# tts_pretrained = 'pretrained/ljspeech_700k_gta.tar.gz'


# DSP --------------------------------------------------------------------------------------------------------------#

# Settings for all models
sample_rate = 22050
n_fft = 2048
fft_bins = n_fft // 2 + 1
num_mels = 80
hop_length = 275                    # 12.5ms - in line with Tacotron 2 paper
win_length = 1100                   # 50ms - same reason as above
fmin = 40
min_level_db = -100
ref_level_db = 20
bits = 9                            # bit depth of signal
mu_law = True                       # Recommended to suppress noise
peak_norm = False                   # Normalise to the peak of each wav file


# WAVERNN / VOCODER ------------------------------------------------------------------------------------------------#


# Model Hparams
voc_upsample_factors = (5, 5, 11)   # NB - this needs to correctly factorise hop_length
voc_rnn_dims = 512
voc_fc_dims = 512
voc_compute_dims = 128
voc_res_out_dims = 128
voc_res_blocks = 10

# Training
voc_batch_size = 32
voc_lr = 1e-4
voc_start_checkpointing = 100_000   # start checkpointing after this amount of steps
voc_checkpoint_every = 25_000
voc_gen_at_checkpoint = 5           # number of samples to generate at each checkpoint
voc_total_steps = 500_000           # Total number of training steps
voc_test_samples = 50               # How many unseen samples to put aside for testing
voc_pad = 2                         # this will pad the input so that the resnet can 'see' wider than input length
voc_seq_len = hop_length * 5        # must be a multiple of hop_length

# Generating / Synthesizing
voc_gen_batched = True              # very fast (realtime+) single utterance batched generation
voc_target = 11_000                 # target number of samples to be generated in each batch entry
voc_overlap = 550                   # number of samples for crossfading between batches


# TACOTRON/TTS -----------------------------------------------------------------------------------------------------#


# Model Hparams
tts_r = 2                           # model predicts r frames per output step
tts_embed_dims = 256                # embedding dimension for the graphemes/phoneme inputs
tts_encoder_dims = 128
tts_decoder_dims = 256
tts_postnet_dims = 128
tts_encoder_K = 16
tts_lstm_dims = 512                 # decoder LSTM dimensions
tts_postnet_K = 8
tts_num_highways = 4
tts_dropout = 0.5
tts_cleaner_names = ['english_cleaners']

# Training
tts_batch_size = 12                 # This fits LJ-Speech into 8GB of GPU mem
tts_lr = 1e-4
tts_max_mel_len = 2000              # if you have a couple of extremely long spectrograms you might want to use this
tts_bin_lengths = True              # bins the spectrogram lengths before sampling in data loader - speeds up training
tts_bin_size = 48                   # how many spectrograms in each bin - must be a multiple of batch size
tts_total_steps = 200_000
tts_clip_grad_norm = 1.0            # clips the gradient norm to prevent explosion
tts_start_checkpoints = 50_000      # start checkpointing after this amount of step
tts_checkpoint_every = 10_000       # checkpoints the model every X steps
tts_plot_every = 1_000              # how often to plot the attention
tts_phoneme_prob = 0.0              # [0 <-> 1] probability for feeding model phonemes vrs graphemes


# ------------------------------------------------------------------------------------------------------------------#

