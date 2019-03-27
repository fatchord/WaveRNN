import glob
from utils.display import *
from utils.dsp import *
import hparams as hp
from multiprocessing import Pool, cpu_count
from utils.paths import Paths
import pickle
import argparse

parser = argparse.ArgumentParser(description='Preprocessing for WaveRNN')
parser.add_argument('--path', '-p', default=hp.wav_path, help='directly point to dataset path (overrides hparams.wav_path')
parser.add_argument('--extension', '-e', default='.wav', help='file extension to search for in dataset folder')
args = parser.parse_args()

extension = args.extension
path = args.path


def get_files(path, extension='.wav') :
    filenames = []
    for filename in glob.iglob(f'{path}/**/*{extension}', recursive=True):
        filenames += [filename]
    return filenames


def convert_file(path) :
    y = load_wav(path)
    peak = np.abs(y).max()
    if peak > 1.0 : y /= peak
    mel = melspectrogram(y)
    quant = encode_mu_law(y, mu=2**hp.bits) if hp.mu_law else float_2_label(y, bits=hp.bits)
    return mel.astype(np.float32), quant.astype(np.int16)


def process_wav(path) :
    id = path.split('/')[-1][:-4]
    m, x = convert_file(path)
    np.save(f'{paths.mel}{id}.npy', m)
    np.save(f'{paths.quant}{id}.npy', x)
    return id


wav_files = get_files(path, extension)
paths = Paths(hp.data_path, hp.model_id)

print(f'\n{len(wav_files)} {extension[1:]} files found in "{path}"\n')

if len(wav_files) == 0 :

    print('Please point wav_path in hparams.py to your dataset,')
    print('or use the --path option.\n')

else :

    simple_table([('Sample Rate', hp.sample_rate),
                  ('Bit Depth', hp.bits),
                  ('Mu Law', hp.mu_law),
                  ('Hop Length', hp.hop_length),
                  ('CPU Count', cpu_count())])

    pool = Pool(processes=cpu_count())
    dataset_ids = []

    for i, id in enumerate(pool.imap_unordered(process_wav, wav_files), 1):
        dataset_ids += [id]
        bar = progbar(i, len(wav_files))
        message = f'{bar} {i}/{len(wav_files)} '
        stream(message)

    with open(f'{paths.data}dataset_ids.pkl', 'wb') as f:
        pickle.dump(dataset_ids, f)

    print('\n\nCompleted. Ready to run "python train.py". \n')
