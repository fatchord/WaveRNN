from utils.dataset import get_datasets
import hparams as hp
from models.fatchord_wavernn import Model
from utils.generation import gen_testset
from utils.paths import Paths
import argparse

parser = argparse.ArgumentParser(description='Generate WaveRNN Samples')
parser.add_argument('--batched', '-b', dest='batched', action='store_true')
parser.add_argument('--unbatched', '-u', dest='batched', action='store_false')
parser.add_argument('--samples', '-s', type=int, help='[int] number of samples to generate')
parser.add_argument('--target', '-t', type=int, help='[int] number of samples in each batch index')
parser.add_argument('--overlap', '-o', type=int, help='[int] number of crossover samples')
parser.set_defaults(batched=hp.batched)
parser.set_defaults(samples=hp.gen_at_checkpoint)
parser.set_defaults(target=hp.target)
parser.set_defaults(overlap=hp.overlap)
args = parser.parse_args()

batched = args.batched
samples = args.samples
target = args.target
overlap = args.overlap

print('\nInitialising Model...\n')

model = Model(rnn_dims=hp.rnn_dims,
              fc_dims=hp.fc_dims,
              bits=hp.bits,
              pad=hp.pad,
              upsample_factors=hp.upsample_factors,
              feat_dims=hp.num_mels,
              compute_dims=hp.compute_dims,
              res_out_dims=hp.res_out_dims,
              res_blocks=hp.res_blocks,
              hop_length=hp.hop_length,
              sample_rate=hp.sample_rate).cuda()

paths = Paths(hp.data_path, hp.model_id)

model.restore(paths.latest_weights)

_, test_set = get_datasets(paths.data)

gen_testset(model, test_set, samples, batched, target, overlap, paths.output)

print('\n\nExiting...\n')
