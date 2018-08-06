# WaveRNN

Pytorch implementation of Deepmind's WaveRNN model from [Efficient Neural Audio Synthesis](https://arxiv.org/abs/1802.08435v1)

<img src="https://raw.githubusercontent.com/fatchord/WaveRNN/master/assets/WaveRNN.png" alt="drawing" width="300px"/>

### Implementation Details
Currently, there are two models in this repo. The first is WaveRNN, however it is quite slow to train (~7 days).

The good news is that I came up with another model that trains much faster and can handle the noise in predicted features from Tacotron and similar TTS models. The sound quality is not as good as Wavenet but it's not that far off. [You can listen to the samples here and judge for yourself.](https://fatchord.github.io/model_outputs/)

### Dependencies
Pytorch v.04
Librosa

**Disclaimer** I do not represent or work for Deepmind/Google.
