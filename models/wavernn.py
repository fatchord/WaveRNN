import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.display import *
from utils.dsp import *


class ResBlock(nn.Module):
    """
    Residual Convolutional Module
    """
    def __init__(self, dims):
        super().__init__()
        self.conv1 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(dims)
        self.batch_norm2 = nn.BatchNorm1d(dims)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        return x + residual


class MelResNet(nn.Module):
    """
    Basic ResNet that prejects melspectrogram to a internal network representation
    """
    def __init__(self, res_blocks, in_dims, compute_dims, res_out_dims):
        super().__init__()
        self.conv_in = nn.Conv1d(in_dims, compute_dims, kernel_size=5, bias=False)
        self.batch_norm = nn.BatchNorm1d(compute_dims)
        self.layers = nn.ModuleList()
        for i in range(res_blocks):
            self.layers.append(ResBlock(compute_dims))
        self.conv_out = nn.Conv1d(compute_dims, res_out_dims, kernel_size=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        for f in self.layers:
            x = f(x)
        x = self.conv_out(x)
        return x


class Stretch2d(nn.Module):
    """
    Filling the missing time steps by copying values
    """
    def __init__(self, x_scale, y_scale):
        super().__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.unsqueeze(-1).unsqueeze(3)
        x = x.repeat(1, 1, 1, self.y_scale, 1, self.x_scale)
        return x.view(b, c, h * self.y_scale, w * self.x_scale)


class UpsampleNetwork(nn.Module):
    """
    Upsampling representation by resizing the input and copying values. Then applies
    conv layers to fine-tune the upsampled representation.
    """
    def __init__(
        self, feat_dims, upsample_scales, compute_dims, res_blocks, res_out_dims, pad
    ):
        super().__init__()
        total_scale = np.cumproduct(upsample_scales)[-1]
        self.indent = pad * total_scale
        self.resnet = MelResNet(res_blocks, feat_dims, compute_dims, res_out_dims)
        self.resnet_stretch = Stretch2d(total_scale, 1)
        self.up_layers = nn.ModuleList()
        for scale in upsample_scales:
            k_size = (1, scale * 2 + 1)
            padding = (0, scale)
            stretch = Stretch2d(scale, 1)
            conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=padding, bias=False)
            conv.weight.data.fill_(1. / k_size[1])
            self.up_layers.append(stretch)
            self.up_layers.append(conv)

    def forward(self, m):
        aux = self.resnet(m).unsqueeze(1)
        aux = self.resnet_stretch(aux)
        aux = aux.squeeze(1)
        m = m.unsqueeze(1)
        for f in self.up_layers:
            m = f(m)
        m = m.squeeze(1)[:, :, self.indent : -self.indent]
        return m.transpose(1, 2), aux.transpose(1, 2)


class Model(nn.Module):
    def __init__(
        self,
        rnn_dims,
        fc_dims,
        bits,
        pad,
        upsample_factors,
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
            feat_dims, upsample_factors, compute_dims, res_blocks, res_out_dims, pad
        )
        self.I = nn.Linear(feat_dims + self.aux_dims + 1, rnn_dims)
        self.rnn1 = nn.GRU(rnn_dims, rnn_dims, batch_first=True)
        self.rnn2 = nn.GRU(rnn_dims + self.aux_dims, rnn_dims, batch_first=True)
        self.fc1 = nn.Linear(rnn_dims + self.aux_dims, fc_dims)
        self.fc2 = nn.Linear(fc_dims + self.aux_dims, fc_dims)
        self.fc3 = nn.Linear(fc_dims, self.n_classes)
        num_params(self)

    def forward(self, x, mels):
        bsize = x.size(0)

        # hidden rnn states
        h1 = torch.zeros(1, bsize, self.rnn_dims).cuda()
        h2 = torch.zeros(1, bsize, self.rnn_dims).cuda()

        # compute input representations from mel-spec.
        mels, aux = self.upsample(mels)

        # split aux features into different temporal segments.
        aux_idx = [self.aux_dims * i for i in range(5)]
        a1 = aux[:, :, aux_idx[0] : aux_idx[1]]
        a2 = aux[:, :, aux_idx[1] : aux_idx[2]]
        a3 = aux[:, :, aux_idx[2] : aux_idx[3]]
        a4 = aux[:, :, aux_idx[3] : aux_idx[4]]

        # concat[mel, aux1] -> FC -> GRU 
        x = torch.cat([x.unsqueeze(-1), mels, a1], dim=2)
        x = self.I(x)
        res = x
        x, _ = self.rnn1(x, h1)

        # concat[x, aux2] -> GRU 
        x = x + res
        res = x
        x = torch.cat([x, a2], dim=2)
        x, _ = self.rnn2(x, h2)

        # concat[x, aux3] -> FC
        x = x + res
        x = torch.cat([x, a3], dim=2)
        x = F.relu(self.fc1(x))

        # concat[x, aux4] -> FC
        x = torch.cat([x, a4], dim=2)
        x = F.relu(self.fc2(x))

        # x -> FC_OUT
        return F.log_softmax(self.fc3(x), dim=-1)

    def preview_upsampling(self, mels):
        mels, aux = self.upsample(mels)
        return mels, aux

    def generate(self, mels):
        self.eval()
        output = []
        rnn1 = self.get_gru_cell(self.rnn1)
        rnn2 = self.get_gru_cell(self.rnn2)

        with torch.no_grad():
            start = time.time()
            x = torch.zeros(1, 1).cuda()
            h1 = torch.zeros(1, self.rnn_dims).cuda()
            h2 = torch.zeros(1, self.rnn_dims).cuda()

            mels = torch.FloatTensor(mels).cuda().unsqueeze(0)
            mels, aux = self.upsample(mels)

            aux_idx = [self.aux_dims * i for i in range(5)]
            a1 = aux[:, :, aux_idx[0] : aux_idx[1]]
            a2 = aux[:, :, aux_idx[1] : aux_idx[2]]
            a3 = aux[:, :, aux_idx[2] : aux_idx[3]]
            a4 = aux[:, :, aux_idx[3] : aux_idx[4]]

            seq_len = mels.size(1)

            for i in tqdm(range(seq_len)):

                m_t = mels[:, i, :]
                a1_t = a1[:, i, :]
                a2_t = a2[:, i, :]
                a3_t = a3[:, i, :]
                a4_t = a4[:, i, :]

                x = torch.cat([x, m_t, a1_t], dim=1)
                x = self.I(x)
                h1 = rnn1(x, h1)

                x = x + h1
                inp = torch.cat([x, a2_t], dim=1)
                h2 = rnn2(inp, h2)

                x = x + h2
                x = torch.cat([x, a3_t], dim=1)
                x = F.relu(self.fc1(x))

                x = torch.cat([x, a4_t], dim=1)
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                posterior = F.softmax(x, dim=1).view(-1)
                distrib = torch.distributions.Categorical(posterior)
                sample = 2 * distrib.sample().float() / (self.n_classes - 1.) - 1.
                output.append(sample)
                x = torch.FloatTensor([[sample]]).cuda()
                if i % 100 == 0:
                    speed = int((i + 1) / (time.time() - start))
        #                     print("{}/{} -- Speed: {} samples/sec".format(i + 1, seq_len, speed))
        output = torch.stack(output).cpu().numpy()
        output = ap.apply_inv_preemphasis(output)
        return output

    def get_gru_cell(self, gru):
        gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
        gru_cell.weight_hh.data = gru.weight_hh_l0.data
        gru_cell.weight_ih.data = gru.weight_ih_l0.data
        gru_cell.bias_hh.data = gru.bias_hh_l0.data
        gru_cell.bias_ih.data = gru.bias_ih_l0.data
        return gru_cell
