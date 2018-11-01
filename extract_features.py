import librosa
import shutil
import argparse
import matplotlib.pyplot as plt
import math, pickle, os, glob
import numpy as np
from tqdm import tqdm
from utils import *
from utils.display import *
from utils.generic_utils import load_config
from utils.audio import AudioProcessor
from multiprocessing import Pool


def get_files(path, extension=".wav"):
    filenames = []
    for filename in glob.iglob(f"{path}/**/*{extension}", recursive=True):
        filenames += [filename]
    return filenames


def convert_file(path):
    wav = ap.load_wav(path, encode=False)
    mel = ap.melspectrogram(wav)
    # quant = (wav + 1.) * (2**bits - 1) / 2
    quant = ap.quantize(wav)
    quant = quant.clip(0, 2 ** CONFIG.audio['bits'] - 1)
    return mel.astype(np.float32), quant.astype(np.int), wav


def process_wav(wav_path):
    idx = wav_path.split("/")[-1][:-4]
    m, x, wav = convert_file(wav_path)
    assert x.max() < 2 ** CONFIG.audio['bits'], wav_path
    assert x.min() >= 0
    np.save(f"{MEL_PATH}{idx}.npy", m)
    np.save(f"{QUANT_PATH}{idx}.npy", x)
    return idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, help="path to config file for feature extraction."
    )
    parser.add_argument(
        "--num_procs", type=int, default=4, help="numer of parallel processes."
    )
    parser.add_argument(
        "--data_path", type=str, default='', help="data path to overwrite config.json."
    )
    
    args = parser.parse_args()

    config_path = args.config_path
    CONFIG = load_config(config_path)

    if args.data_path != '':
        CONFIG.data_path = args.data_path

    ap = AudioProcessor(**CONFIG.audio)

    # Point SEG_PATH to a folder containing your training wavs
    # Doesn't matter if it's LJspeech, CMU Arctic etc. it should work fine
    SEG_PATH = CONFIG.data_path
    OUT_PATH = os.path.join(CONFIG.out_path, CONFIG.run_name, "data/")
    QUANT_PATH = os.path.join(OUT_PATH, "quant/")
    MEL_PATH = os.path.join(OUT_PATH, "mel/")
    os.makedirs(OUT_PATH, exist_ok=True)
    os.makedirs(QUANT_PATH, exist_ok=True)
    os.makedirs(MEL_PATH, exist_ok=True)

    wav_files = get_files(SEG_PATH)
    print(" > Number of audio files : {}".format(len(wav_files)))

    wav_file = wav_files[1]
    m, x, wav = convert_file(wav_file)

    # save an example for sanity check
    x = ap.dequantize(x)
    librosa.output.write_wav(
        OUT_PATH + "test_converted_audio.wav", x, sr=CONFIG.audio['sample_rate']
    )
    shutil.copyfile(wav_files[1], OUT_PATH + "test_target_audio.wav")

    # This will take a while depending on size of dataset
    with Pool(8) as p:
        dataset_ids = list(tqdm(p.imap(process_wav, wav_files), total=len(wav_files)))

    # save metadata
    with open(os.path.join(OUT_PATH, "dataset_ids.pkl"), "wb") as f:
        pickle.dump(dataset_ids, f)
