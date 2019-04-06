import os


class Paths :
    def __init__(self, data_path, voc_id, tts_id) :
        # Data Paths
        self.data = f'{data_path}/{voc_id}/'
        self.quant = f'{self.data}quant/'
        self.mel = f'{self.data}mel/'
        self.gta = f'{self.data}gta/'
        # WaveRNN/Vocoder Paths
        self.voc_checkpoints = f'checkpoints/{voc_id}.wavernn/'
        self.voc_latest_weights = f'{self.voc_checkpoints}latest_weights.pyt'
        self.voc_output = f'model_outputs/{voc_id}.wavernn/'
        self.voc_step = f'{self.voc_checkpoints}/step.npy'
        self.voc_log = f'{self.voc_checkpoints}log.txt'
        # Tactron/TTS Paths
        self.tts_checkpoints = f'checkpoints/{tts_id}.tacotron/'
        self.tts_latest_weights = f'{self.tts_checkpoints}latest_weights.pyt'
        self.tts_output = f'model_outputs/{tts_id}.tts/'
        self.tts_step = f'{self.tts_checkpoints}/step.npy'
        self.tts_log = f'{self.tts_checkpoints}log.txt'
        self.tts_attention = f'{self.tts_checkpoints}/attention/'
        self.tts_mel_plot = f'{self.tts_checkpoints}/mel_plots/'
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
        os.makedirs(self.tts_attention, exist_ok=True)
        os.makedirs(self.tts_mel_plot, exist_ok=True)

