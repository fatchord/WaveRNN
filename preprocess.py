import glob
from utils.display import *
from utils.dsp import *
import hparams as hp
from multiprocessing import Pool, cpu_count
from utils.paths import Paths
import pickle


def get_files(path, extension='.wav') :
    filenames = []
    for filename in glob.iglob(f'{path}/**/*{extension}', recursive=True):
        filenames += [filename]
    return filenames


def convert_file(path) :
    wav = load_wav(path)
    mel = melspectrogram(wav)
    if hp.mu_law :
        quant = encode_mu_law(wav, mu=2 ** hp.bits)
    else :
        quant = float_2_label(wav, bits=hp.bits)
    return mel.astype(np.float32), quant.astype(np.int16)


def process_wav(path) :
    id = path.split('/')[-1][:-4]
    m, x = convert_file(path)
    np.save(f'{paths.mel}{id}.npy', m)
    np.save(f'{paths.quant}{id}.npy', x)
    return id


wav_files = get_files(hp.wav_path)
paths = Paths(hp.data_path, hp.model_id)

print(f'\n{len(wav_files)} wav files found in hparams.wav_path\n')

if len(wav_files) == 0 :
    print('Please point wav_path in hparams.py to your dataset\n')

else :

    print('+--------------------+--------------+---------+-----------------+')
    print(f'| Sample Rate: {hp.sample_rate} | Mu Law: {hp.mu_law} | Bits: {hp.bits} | Hop Length: {hp.hop_length} |')
    print('+--------------------+--------------+---------+-----------------+')

    pool = Pool(processes=cpu_count())
    dataset_ids = []

    print(f'\nCPU count: {cpu_count()}\n')

    for i, id in enumerate(pool.imap_unordered(process_wav, wav_files), 1):
        dataset_ids += [id]
        bar = progbar(i, len(wav_files))
        message = f'{bar} {i}/{len(wav_files)} '
        stream(message)

    with open(f'{paths.data}dataset_ids.pkl', 'wb') as f:
        pickle.dump(dataset_ids, f)

    print('\n\nCompleted. Read to run "python train.py". \n')
