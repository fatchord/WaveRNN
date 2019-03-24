import os


class Paths :
    def __init__(self, data_path, model_id) :
        self.data = data_path
        self.quant = f'{data_path}/quant/'
        self.mel = f'{data_path}/mel/'
        self.checkpoints = f'checkpoints/{model_id}/'
        self.latest_weights = f'{self.checkpoints}latest_weights.pyt'
        self.output = f'model_outputs/{model_id}/'
        self.step = f'{self.checkpoints}/step.npy'
        self.log = f'{self.checkpoints}log.txt'
        self.create_paths()

    def create_paths(self) :
        os.makedirs(self.data, exist_ok=True)
        os.makedirs(self.quant, exist_ok=True)
        os.makedirs(self.mel, exist_ok=True)
        os.makedirs(self.checkpoints, exist_ok=True)
        os.makedirs(self.output, exist_ok=True)
