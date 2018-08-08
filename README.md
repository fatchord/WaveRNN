# WaveRNN

Pytorch implementation of Deepmind's WaveRNN model from [Efficient Neural Audio Synthesis](https://arxiv.org/abs/1802.08435v1)

<img src="https://raw.githubusercontent.com/fatchord/WaveRNN/master/assets/WaveRNN.png" alt="drawing" width="600px"/>
<img src="https://raw.githubusercontent.com/fatchord/WaveRNN/master/assets/wavernn_alt_model_hrz2.png" alt="drawing" width="600px"/>

### Implementation Details
Currently, there are two models in this repo. The first is WaveRNN, however it is quite slow to train (~7 days).

The good news is that I came up with another model that trains much faster and can handle the noise in predicted features from Tacotron and similar models. The sound quality is not as good as Wavenet but it's not that far off. [You can listen to the samples here and judge for yourself.](https://fatchord.github.io/model_outputs/)

Notebooks 1 - 4 are self-contained however notebooks 5a and 5b need to be run sequentially. You can stop & close notebook 5b (training) whenever you like and it will pick up from where you left off.


### Dependencies
* Python 3
* Pytorch v.04
* Librosa

**Disclaimer** I do not represent or work for Deepmind/Google.
