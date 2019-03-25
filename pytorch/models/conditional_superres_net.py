import torch
import torch.nn as nn
from pytorch.models.conditioning_nlcd import Conditioning_nlcd
import json, os
import torch.nn.functional as F
import utils.model_utils as nn_utils

"""
@uthor: Anthony Ortiz
Date: 03/25/2019
Last Modified: 03/25/2019
"""

class Down(nn.Module):
    """
    Down blocks in U-Net
    """
    def __init__(self, conv, max):
        super(Down, self).__init__()
        self.conv = conv
        self.max = max

    def forward(self, x, gamma, beta):
        x = self.conv(x, gamma, beta)
        return self.max(x), x, x.shape[2]


class Up(nn.Module):
    """
    Up blocks in U-Net

    Similar to the down blocks, but incorporates input from skip connections.
    """
    def __init__(self, up, conv):
        super(Up, self).__init__()
        self.conv = conv
        self.up = up

    def forward(self, x, conv_out, D, gamma, beta):
        x = self.up(x)
        lower = int(0.5 * (D - x.shape[2]))
        upper = int(D - lower)
        conv_out_ = conv_out[:, :, lower:upper, lower:upper] # adjust to zero padding
        x = torch.cat([x, conv_out_], dim=1)
        return self.conv(x, gamma, beta)

class Conditional_superres_net(nn.Module):
    def __init__(self, model_opts, n_embedding_units=9 * 9 * 64):

        super(Conditional_superres_net, self).__init__()
        self.opts = model_opts["conditional_superres_net_opts"]
        self.n_input_channels = self.opts["n_input_channels"]
        self.n_classes = self.opts["n_classes"]

        self.n_embedding_units_cbn = n_embedding_units
        self.n_hidden_cbn = self.opts["n_hidden_cbn"]
        self.n_features_cbn = self.opts["n_features_cbn"]

        #this predict latlong
        self.conditioning_model = Conditioning_nlcd(model_opts)
        self.conditioning_model.train()



        # MLP used to predict betas and gammas
        self.fc_cbn = nn.Sequential(
            nn.Linear(self.n_embedding_units_cbn, self.n_hidden_cbn),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_hidden_cbn, 2*self.n_features_cbn),
        )
        #U-Net

        # down transformations
        max2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_1 = Down(C_UNet_conditioning_block(model_opts, self.n_input_channels, 32), max2d)
        self.down_2 = Down(C_UNet_conditioning_block(model_opts, 32, 64), max2d)
        self.down_3 = Down(C_UNet_conditioning_block(model_opts, 64, 128), max2d)
        self.down_4 = Down(C_UNet_conditioning_block(model_opts, 128, 256), max2d)

        # midpoint
        self.conv5_block = C_UNet_conditioning_block(model_opts, 256, 512)

        # up transformations
        conv_tr = lambda x, y: nn.ConvTranspose2d(x, y, kernel_size=2, stride=2)
        self.up_1 = Up(conv_tr(512, 256), C_UNet_conditioning_block(model_opts, 512, 256))
        self.up_2 = Up(conv_tr(256, 128), C_UNet_conditioning_block(model_opts, 256, 128))
        self.up_3 = Up(conv_tr(128, 64), C_UNet_conditioning_block(model_opts, 128, 64))
        self.up_4 = Up(conv_tr(64, 32), C_UNet_conditioning_block(model_opts, 64, 32))

        # Final output
        self.conv_final = nn.Conv2d(in_channels=32, out_channels=self.n_classes,
                                    kernel_size=1, padding=0, stride=1)

    def forward(self, x):

        conditioning_pred = self.conditioning_model(x)
        conditioning_info = self.conditioning_model.pre_pred(x)

        cbn = self.fc_cbn(conditioning_info)
        gammas = cbn[:, :int(cbn.shape[1]/2)]
        betas = cbn[:, int(cbn.shape[1]/2):]

        # down layers
        x, conv1_out, conv1_dim = self.down_1(x, gammas[:, :32], betas[:, :32])
        x, conv2_out, conv2_dim = self.down_2(x, gammas[:, 32:96], betas[:, 32:96])
        x, conv3_out, conv3_dim = self.down_3(x, gammas[:, 96:224], betas[:, 96:224])
        x, conv4_out, conv4_dim = self.down_4(x, gammas[:, 224:480], betas[:, 224:480])

        # Bottleneck
        x = self.conv5_block(x, gammas[:, 480:992], betas[:, 480:992])

        # up layers
        x = self.up_1(x, conv4_out, conv4_dim, gammas[:, 992:1248], betas[:, 992:1248])
        x = self.up_2(x, conv3_out, conv3_dim, gammas[:, 1248:1376], betas[:, 1248:1376])
        x = self.up_3(x, conv2_out, conv2_dim, gammas[:, 1376:1440], betas[:, 1376:1440])
        x = self.up_4(x, conv1_out, conv1_dim, gammas[:, 1440:1472], betas[:, 1440:1472])

        #Output
        x = self.conv_final(x)

        if self.opts["end_to_end"]:
            return x, conditioning_pred

        return x


class C_UNet_conditioning_block(nn.Module):
    def __init__(self, model_opts,  dim_in, dim_out):
        super().__init__()

        self.opts = model_opts["fully_conditional_unet_opts"]
        self.conv_block1 = self.conv_block(dim_in, dim_out)

    def conv_block(self, dim_in, dim_out, kernel_size=3, stride=1, padding=0, bias=True):
        """
        This is the main conv block for Unet. Two conv2d
        :param dim_in:
        :param dim_out:
        :param kernel_size:
        :param stride:
        :param padding:
        :param bias:
        :param useBN:
        :param useGN:
        :return:
        """
        if self.opts["conditioning_type"] == "CBN":
            return nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(dim_out, affine=False),
            )
        elif self.opts["conditioning_type"] == "CGN":
            return nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn_utils.GroupNorm(dim_out)
            )
        elif self.opts["conditioning_type"] == "CN":
            return nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            )
        else:
            print("Conditioning type {} not supported. Available options: CGN, CBN".format(self.opts["conditioning_type"]))
            raise NotImplementedError

    def forward(self, input, gamma, beta):
        out = self.conv_block1(input)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        out = gamma * out + beta

        out = F.relu(out)

        return out


#Test with mock data
if __name__ == "__main__":
    # A full forward pass
    params = json.load(open(os.environ["PARAMS_PATH"], "r"))
    model_opts = params["model_opts"]
    im = torch.randn(1, 4, 240, 240)
    model = Conditional_superres_net(model_opts)
    x, latlon_pred = model(im)
    #x = model(im)
    print(x.shape)
    del model
    del x