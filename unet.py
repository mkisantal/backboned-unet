import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F

def get_backbone(name, pretrained=True):

    """ Loading backbone, defining names for skip-connections and encoder output. """

    # TODO: More backbones

    if name == 'resnet18':
        backbone = models.resnet50(pretrained=pretrained)
    elif name == 'resnet34':
        backbone = models.resnet34(pretrained=pretrained)
    elif name == 'resnet50':
        backbone = models.resnet50(pretrained=pretrained)
    elif name == 'resnet101':
        backbone = models.resnet101(pretrained=pretrained)
    elif name == 'resnet152':
        backbone = models.resnet152(pretrained=pretrained)
    elif name == 'vgg16':
        backbone = models.vgg16_bn(pretrained=pretrained).features  # without fc head
    elif name == 'vgg19':
        backbone = models.vgg19_bn(pretrained=pretrained).features  # without fc head
    else:
        raise NotImplemented('Only resnet models implemented so far.')

    if name.startswith('resnet'):
        feature_names = [None, 'relu', 'layer1', 'layer2', 'layer3']
        backbone_output = 'layer4'
    elif name == 'vgg16':
        # TODO: consider using a 'bridge' for VGG models, there is just a MaxPool between last skip and backbone output
        feature_names = ['5', '12', '22', '32', '42']
        backbone_output = '43'
    elif name == 'vgg19':
        feature_names = ['5', '12', '25', '38', '51']
        backbone_output = '52'
    else:
        raise NotImplemented('Only resnet models implemented so far.')

    return backbone, feature_names, backbone_output


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

    """ U-Net (https://arxiv.org/pdf/1505.04597.pdf) implementation with pre-trained torchvision backbones."""

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

        self.backbone_name = backbone_name

        self.backbone, self.shortcut_features, self.bb_out_name = get_backbone(backbone_name)
        shortcut_chs, bb_out_chs = self.infer_skip_channels()
        print(shortcut_chs)
        # build decoder part
        # TODO: build with loop?
        self.upsample_blocks = nn.ModuleList()
        self.upsample_blocks.append(UpsampleBlock(bb_out_chs, 256, skip_in=shortcut_chs[4], parametric=True))
        self.upsample_blocks.append(UpsampleBlock(256, 128, skip_in=shortcut_chs[3], parametric=True))
        self.upsample_blocks.append(UpsampleBlock(128, 64, skip_in=shortcut_chs[2], parametric=True))
        self.upsample_blocks.append(UpsampleBlock(64, 32, skip_in=shortcut_chs[1], parametric=True))
        self.upsample_blocks.append(UpsampleBlock(32, 16, skip_in=shortcut_chs[0], parametric=True))

        self.final_conv = nn.Conv2d(16, classes, kernel_size=(1, 1))

        # TODO: optionally freeze encoder weights

    def forward(self, *input):
        if self.backbone_name.startswith('resnet'):
            x, features = self.forward_resnet_backbone(*input)
        else:
            x, features = self.forward_vgg_backbone(*input)

        print('Backbone ran.')
        print('x: {}'.format(x.shape))
        for skip_name in self.shortcut_features:
            if features[skip_name] is not None:
                print('\t[{}]: {}'.format(skip_name, features[skip_name].shape))
            else:
                print('\t[{}]: {}'.format(skip_name, features[skip_name]))
        for skip_name, upsample_block in zip(self.shortcut_features[::-1], self.upsample_blocks):
            print(skip_name)
            skip_features = features[skip_name]
            x = upsample_block(x, skip_features)

        x = self.final_conv(x)
        return x

    def forward_vgg_backbone(self, x):
        features = {None: None}
        for name, child in self.backbone.named_children():
            x = child(x)

            if name in self.shortcut_features:
                features[name] = x

        return x, features

    def forward_resnet_backbone(self, x):

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

        return x, features

    def infer_skip_channels(self):

        """ Getting the number of channels at skip connections and at the output of the encoder. """

        x = torch.zeros(1, 3, 224, 224)
        channels = [0] if self.backbone_name.startswith('resnet') else []  # no features at full resolution for resnet

        # forward run in backbone to count channels
        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                channels.append(x.shape[1])
            if name == self.bb_out_name:
                out_channels = x.shape[1]
                break
        return channels, out_channels


if __name__ == "__main__":

    # simple test run
    net = Unet(backbone_name='vgg19')

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters())
    print('Network initialized. Running a test batch.')
    for _ in range(1):
        with torch.set_grad_enabled(True):
            batch = torch.empty(1, 3, 224, 224).normal_()
            targets = torch.empty(1, 1, 224, 224).normal_()

            out = net(batch)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
        print(out.shape)

    print('fasza.')
