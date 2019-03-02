import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import time


class UpsampleNetwork(nn.Module):
    """
    Upsampling representation by resizing the input and copying values. Then applies
    conv layers to fine-tune the upsampled representation.
    """
    def __init__(
        self, feat_dims, upsample_scale, compute_dims, res_blocks, res_out_dims
    ):
        super().__init__()
        self.up_layer = nn.Upsample(scale_factor=upsample_scale, mode='linear', align_corners=True)

    def forward(self, m):
        m = self.up_layer(m)
        return m.transpose(1, 2)


class Model(nn.Module):
    def __init__(
        self,
        rnn_dims,
        fc_dims,
        bits,
        upsample_factor,
        feat_dims,
        compute_dims,
        res_out_dims,
        res_blocks,
    ):
        super().__init__()
        self.n_classes = 2 ** bits
        self.rnn_dims = rnn_dims
        self.aux_dims = res_out_dims // 4
        self.upsample = UpsampleNetwork(
            feat_dims, upsample_factor, compute_dims, res_blocks, res_out_dims
        )
        self.I = nn.Linear(feat_dims + 1, rnn_dims)
        self.rnn1 = nn.GRU(rnn_dims, rnn_dims, batch_first=True)
        self.rnn2 = nn.GRU(rnn_dims, rnn_dims, batch_first=True)
        self.fc1 = nn.Linear(rnn_dims, fc_dims)
        self.fc2 = nn.Linear(fc_dims, fc_dims)
        self.fc3 = nn.Linear(fc_dims, self.n_classes)

    def forward(self, x, mels):
        bsize = mels.size(0)

        # hidden rnn states     
        h1 = x.data.new(1, bsize, self.rnn_dims).zero_()
        h2 = x.data.new(1, bsize, self.rnn_dims).zero_()

        # compute input representations from mel-spec.
        mels = self.upsample(mels)
        assert mels.shape[1] == x.shape[1],'{} vs {}'.format(mels.shape[1], x.shape[1])

        # concat[mel, aux1] -> FC -> GRU 
        x = torch.cat([x.unsqueeze(-1), mels], dim=2)
        x = self.I(x)
        res = x
        x, _ = self.rnn1(x, h1)

        # concat[x, aux2] -> GRU 
        x = x + res
        res = x
        x, _ = self.rnn2(x, h2)

        # concat[x, aux3] -> FC
        x = x + res
        x = F.relu(self.fc1(x))

        # concat[x, aux4] -> FC
        x = F.relu(self.fc2(x))

        # x -> FC_OUT
        return F.log_softmax(self.fc3(x), dim=-1)

    def preview_upsampling(self, mels):
        mels, aux = self.upsample(mels)
        return mels, aux

    def generate(self, mels, verbose=False, deterministic=True, use_cuda=False):
        output = []
        rnn1 = self.get_gru_cell(self.rnn1)
        rnn2 = self.get_gru_cell(self.rnn2)

        with torch.no_grad():
            start = time.time()
            x = mels.data.new(1, 1).zero_()
            h1 = mels.data.new(1, self.rnn_dims).zero_()
            h2 = mels.data.new(1, self.rnn_dims).zero_()

            mels = self.upsample(mels)

            seq_len = mels.size(1)

            for i in range(seq_len):

                m_t = mels[:, i, :]

                x = torch.cat([x, m_t], dim=1)
                x = self.I(x)
                h1 = rnn1(x, h1)

                x = x + h1
                h2 = rnn2(x, h2)

                x = x + h2
                x = F.relu(self.fc1(x))

                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                if deterministic:
                    sample = torch.argmax(F.softmax(x, dim=1).view(-1), dim=-1)
                    sample = 2 * sample.float() / (self.n_classes - 1.) - 1.
                else:
                    posterior = F.softmax(x, dim=1).view(-1)
                    distrib = torch.distributions.Categorical(posterior)
                    sample = 2 * distrib.sample().float() / (self.n_classes - 1.) - 1.

                output.append(sample)
                x = torch.FloatTensor([[sample]]).to(mels.device)
                if i % 1000 == 0:
                    speed = int((i + 1) / (time.time() - start))
                    if verbose:
                        print("{}/{} -- Speed: {} samples/sec".format(i + 1, seq_len, speed))
        output = torch.stack(output).cpu().numpy()
        return output

    def get_gru_cell(self, gru):
        gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
        gru_cell.weight_hh.data = gru.weight_hh_l0.data
        gru_cell.weight_ih.data = gru.weight_ih_l0.data
        gru_cell.bias_hh.data = gru.bias_hh_l0.data
        gru_cell.bias_ih.data = gru.bias_ih_l0.data
        return gru_cell
