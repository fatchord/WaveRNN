# CONFIG
wav_path = '/path/to/wav_files/'
data_path = 'data/'
model_id = '9bit_mulaw'


# DSP
sample_rate = 22050
n_fft = 2048
fft_bins = n_fft // 2 + 1
num_mels = 80
hop_length = 275                # 12.5ms - in line with Tacotron 2 paper
win_length = 1100               # 50ms - same as above
fmin = 40
min_level_db = -100
ref_level_db = 20
bits = 9                        # bit depth of signal
mu_law = True                   # Recommended to suppress noise
peak_norm = False

# MODEL
upsample_factors = (5, 5, 11)   # NB - this needs to correctly factorise hop_length - triple-check if modifying
rnn_dims = 512
fc_dims = 512
compute_dims = 128
res_out_dims = 128
res_blocks = 10


# TRAINING
batch_size = 32
lr = 1e-4
checkpoint_every = 25_000
gen_at_checkpoint = 5           # number of samples to generate at each checkpoint
total_steps = 500_000           # Total number of training steps
test_samples = 50               # How many unseen samples to put aside for testing
pad = 2                         # this will pad the input so that the resnet conv can 'see' wider than input length
seq_len = hop_length * 5        # must be a multiple of hop_length


# GENERATING
batched = True                  # very fast (realtime+) single utterance batched generation
target = 11_000                 # target number of samples to be generated in each batch entry
overlap = 550                   # number of samples for crossfading between batches