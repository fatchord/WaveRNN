import hparams as hp
from utils.dsp import *

def gen_testset(model, test_set, samples, batched, target, overlap, save_path) :

    k = model.get_step() // 1000

    for i, (m, x) in enumerate(test_set, 1):

        if i > samples : break

        print('\n| Generating: %i/%i' % (i, samples))

        x = x[0].numpy()

        if hp.mu_law :
            x = decode_mu_law(x, 2**hp.bits, from_labels=True)
        else :
            x = label_2_float(x, hp.bits)

        save_wav(x, f'{save_path}{k}k_steps_{i}_target.wav')

        batch_str = f'gen_batched_target{target}_overlap{overlap}' if batched else 'gen_NOT_BATCHED'
        save_str = f'{save_path}{k}k_steps_{i}_{batch_str}.wav'

        _ = model.generate(m, save_str, batched, target, overlap, hp.mu_law)
