# CONFIG
wav_path = '/path/to/wav_files/'
data_path = 'data/'
model_id = '9bits_linear'


# DSP
sample_rate = 22050
n_fft = 2048
fft_bins = n_fft // 2 + 1
num_mels = 80
hop_length = 275  # 12.5ms
upsample_factors = (5, 5, 11)  # NB - this needs to correctly factorise hop_length
win_length = 1100  # 50ms
fmin = 40
min_level_db = -100
ref_level_db = 20
bits = 9



# MODEL
rnn_dims = 512
fc_dims = 512
feat_dims = 80
compute_dims = 128
res_out_dims = 128
res_blocks = 10


# TRAINING
checkpoint_every = 25_000
total_steps = 300_000
batch_size = 32
test_samples = 50
lr = 1e-4
pad = 2
seq_len = hop_length * 5  # must be a multiple of hop_length
