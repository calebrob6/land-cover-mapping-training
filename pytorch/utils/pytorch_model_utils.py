import torch
import torch.nn as nn
import shutil, os

"""
@uthor: Anthony Ortiz
Date: 03/25/2019
Last Modified: 03/25/2019
"""
class GroupNorm(nn.Module):
    def __init__(self, num_features, channels_per_group=8, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1,num_features,1,1))
        self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))
        self.channels_per_group = channels_per_group
        self.eps = eps

    def forward(self, x):
        N,C,H,W = x.size()
        G = int(C/self.channels_per_group)
        assert C % G == 0

        x = x.view(N,G,-1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N,C,H,W)
        return x * self.weight + self.bias


class CheckpointSaver(object):
    def __init__(self, save_dir, backup_dir):
        self.save_dir = save_dir
        self.backup_dir = backup_dir

    def save(self, state, is_best, checkpoint_name='checkpoint'):
        checkpoint_path = os.path.join(self.save_dir,
                                       '{}.pth.tar'.format(checkpoint_name))
        try:
            shutil.copyfile(
                checkpoint_path,
                '{}_bak'.format(checkpoint_path)
            )
        except IOError:
            pass
        torch.save(state, checkpoint_path)
        if is_best:
            try:
                shutil.copyfile(
                    os.path.join(self.backup_dir,
                                '{}_best.pth.tar'.format(checkpoint_name)),
                    os.path.join(self.backup_dir,
                                '{}_best.pth.tar_bak'.format(checkpoint_name))
                )
            except IOError:
                pass
            shutil.copyfile(
                checkpoint_path,
                os.path.join(self.backup_dir,
                             '{}_best.pth.tar'.format(checkpoint_name))
            )

class GroupNormNN(nn.Module):
    def __init__(self, num_features, channels_per_group=8, window_size=(32,32), eps=1e-5):
        super(GroupNormNN, self).__init__()
        self.weight = nn.Parameter(torch.ones(1,num_features,1,1))
        self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))
        self.channels_per_group = channels_per_group
        self.eps = eps
        self.window_size = window_size


    def forward(self, x):
        N,C,H,W = x.size()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        G = int(C/self.channels_per_group)
        assert C % G == 0
        if self.window_size[0] < H and self.window_size[1]<W:
            with torch.no_grad():
                x_new = torch.unsqueeze(x, dim=1)
                weights1 = torch.ones((1, 1, self.channels_per_group,1, self.window_size[1])).to(device)
                weights2 = torch.ones((1,1, 1, self.window_size[0],1)).to(device)
                sums1 = F.conv3d(x_new, weights1, stride=[self.channels_per_group, 1, 1])
                sums = F.conv3d(sums1, weights2)
                x_squared = x_new * x_new
                squares1 = F.conv3d(x_squared, weights1, stride=[self.channels_per_group, 1, 1])
                squares = F.conv3d(squares1, weights2)

                n = self.window_size[0] * self.window_size[1] * self.channels_per_group
                means = torch.squeeze((sums / n), dim=1)
                var = torch.squeeze((1.0 / n * (squares - sums * sums / n)), dim=1)
                _,_, r,c = means.size()

                pad2d =(int(math.floor((W- c)/2)), int(math.ceil((W- c)/2)), int(math.floor((H- r)/2)), int(math.ceil((H- r)/2)))
                padded_means = F.pad(means, pad2d, 'replicate')
                padded_vars = F.pad(var, pad2d, 'replicate')

            for i in range(G):
                x[:, i * self.channels_per_group:i * self.channels_per_group + self.channels_per_group, :, :] = (x[:,
                                                                                                                 i * self.channels_per_group:i * self.channels_per_group + self.channels_per_group,
                                                                                                                 :,
                                                                                                                 :] - torch.unsqueeze(
                    padded_means[:, i, :, :], dim=1).to(device)) / (torch.unsqueeze(padded_vars[:, i, :, :], dim=1).to(
                    device) + self.eps).sqrt()

        else:
            x = x.view(N, G, -1)
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True)

            x = (x - mean) / (var + self.eps).sqrt()
            x = x.view(N, C, H, W)

        return x * self.weight + self.bias