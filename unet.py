import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F


def get_backbone(name, pretrained=True):

    # TODO: More backbones

    if name == 'resnet50':
        backbone = models.resnet50(pretrained=pretrained)
    return backbone


class UpsampleBlock(nn.Module):

    # TODO: separate parametric and non-parametric classes?
    # TODO: skip connection concatenated OR added

    def __init__(self, ch_in, ch_out=None, skip_in=0, use_bn=True, parametric=False):
        super(UpsampleBlock, self).__init__()

        self.parametric = parametric
        ch_out = ch_in/2 if ch_out is None else ch_out

        # first convolution: either transposed conv, or conv following the skip connection
        if parametric:
            # versions: kernel=4 padding=1, kernel=2 padding=0
            self.up = nn.ConvTranspose2d(in_channels=ch_in, out_channels=ch_out, kernel_size=(4, 4),
                                         stride=2, padding=1, output_padding=0, bias=(not use_bn))
            self.bn1 = nn.BatchNorm2d(ch_out) if use_bn else None
        else:
            self.up = None
            ch_in = ch_in + skip_in
            self.conv1 = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=(3, 3),
                                   stride=1, padding=1, bias=(not use_bn))
            self.bn1 = nn.BatchNorm2d(ch_out) if use_bn else None

        self.relu = nn.ReLU(inplace=True)

        # second convolution
        conv2_in = ch_out if not parametric else ch_out + skip_in
        self.conv2 = nn.Conv2d(in_channels=conv2_in, out_channels=ch_out, kernel_size=(3, 3),
                               stride=1, padding=1, bias=(not use_bn))
        self.bn2 = nn.BatchNorm2d(ch_out) if use_bn else None

    def forward(self, x, skip_connection=None):

        x = self.up(x) if self.parametric else F.interpolate(x, size=None, scale_factor=2, mode='bilinear',
                                                             align_corners=None)
        if self.parametric:
            x = self.bn1(x) if self.bn1 is not None else x
            x = self.relu(x)

        if skip_connection is not None:
            x = torch.cat([x, skip_connection], dim=1)

        if not self.parametric:
            x = self.conv1(x)
            x = self.bn1(x) if self.bn1 is not None else x
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x) if self.bn2 is not None else x
        x = self.relu(x)

        return x


class Unet(nn.Module):

    def __init__(self,
                 backbone_name='resnet50',
                 input_shape=(None, None, 3),
                 classes=1,
                 activation='sigmoid',
                 encoder_weights='imagenet',
                 encoder_freeze=False,
                 encoder_features='default',
                 decoder_block_type='upsampling',
                 decoder_filters=(256, 128, 64, 32, 16),
                 decoder_use_batchnorm=True):
        super(Unet, self).__init__()

        self.backbone = get_backbone(backbone_name)

        # extract feature layers from backbone
        # TODO: defining feature layers for different networks
        self.shortcut_features = [None, 'relu', 'layer1', 'layer2', 'layer3']

        # build decoder part
        # TODO: build with loop?
        self.upsample_blocks = nn.ModuleList()
        self.upsample_blocks.append(UpsampleBlock(2048, 256, skip_in=1024, parametric=True))
        self.upsample_blocks.append(UpsampleBlock(256, 128, skip_in=512, parametric=True))
        self.upsample_blocks.append(UpsampleBlock(128, 64, skip_in=256, parametric=True))
        self.upsample_blocks.append(UpsampleBlock(64, 32, skip_in=64, parametric=True))
        self.upsample_blocks.append(UpsampleBlock(32, 16, skip_in=0, parametric=True))

        self.final_conv = nn.Conv2d(16, classes, kernel_size=(1, 1))

        # TODO: optionally freeze encoder weights

    def forward(self, x):

        features = {None: None}

        # running encoder
        for name, child in self.backbone.named_children():
            if name == 'fc':
                x = x.view(x.size(0), -1)
            # actual forward
            x = child(x)

            # storing intermediate results
            if name in self.shortcut_features:
                features[name] = x

            if name == 'layer4':
                break

        # running decoder
        for skip_name, upsample_block in zip(self.shortcut_features[::-1], self.upsample_blocks):
            skip_features = features[skip_name]
            x = upsample_block(x, skip_features)

        x = self.final_conv(x)

        # interact(local=locals())

        return x


if __name__ == "__main__":

    # simple test run
    net = Unet().cuda()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters())

    for _ in range(1):
        with torch.set_grad_enabled(True):

            batch = torch.empty(3, 3, 224, 224).normal_().cuda()
            targets = torch.empty(3, 1, 224, 224).normal_().cuda()

            out = net(batch)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
        print(out.shape)

    print('fasza.')
