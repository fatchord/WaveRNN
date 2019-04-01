import os


class Paths :
    def __init__(self, data_path, model_id) :
        # Data Paths
        self.data = data_path
        self.quant = f'{data_path}/quant/'
        self.mel = f'{data_path}/mel/'
        self.gta = f'{data_path}/gta/'
        # WaveRNN/Vocoder Paths
        self.voc_checkpoints = f'checkpoints/{model_id}.vocoder/'
        self.voc_latest_weights = f'{self.voc_checkpoints}latest_weights.pyt'
        self.voc_output = f'model_outputs/{model_id}.vocoder/'
        self.voc_step = f'{self.voc_checkpoints}/step.npy'
        self.voc_log = f'{self.voc_checkpoints}log.txt'
        # Tactron/TTS Paths
        self.tts_checkpoints = f'checkpoints/{model_id}.tacotron/'
        self.tts_latest_weights = f'{self.tts_checkpoints}latest_weights.pyt'
        self.tts_output = f'model_outputs/{model_id}.tts/'
        self.tts_step = f'{self.tts_checkpoints}/step.npy'
        self.tts_log = f'{self.tts_checkpoints}log.txt'
        self.create_paths()

    def create_paths(self) :
        os.makedirs(self.data, exist_ok=True)
        os.makedirs(self.quant, exist_ok=True)
        os.makedirs(self.mel, exist_ok=True)
        os.makedirs(self.gta, exist_ok=True)
        os.makedirs(self.voc_checkpoints, exist_ok=True)
        os.makedirs(self.voc_output, exist_ok=True)
        os.makedirs(self.tts_checkpoints, exist_ok=True)
        os.makedirs(self.tts_output, exist_ok=True)
