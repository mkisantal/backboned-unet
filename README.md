# backboned-unet
U-Nets for image segmentation with pre-trained backbones in PyTorch.

## Why another U-Net implementation?
I was looking for an U-Net PyTorch implementation that can use pre-trained
torchvision models as backbones in the encoder path. There is a great
[repo](https://github.com/qubvel/segmentation_models)
for this in Keras, but I didn't find a good PyTorch implementation that works
with multiple torchvision models. So I decided to create one.

### WIP

So far VGG, ResNet and DenseNet backbones have been implemented.

### Setup

Installing package:

    git clone https://github.com/mkisantal/backboned-unet.git
    cd backboned-unet
    pip install .

### Simple usage example
The U-net model can be imported just like any other torchvision model. The user can specify
a backbone architecture, choose upsampling operation (transposed convolution or bilinear upsampling
followed by convolution), specify the number of filters in the different decoder stages, etc.

    from backboned_unet import Unet
    net = Unet(backbone_name='densenet121', classes=21)
    
The module loads the backbone torchvision model, and builds a decoder on top of it using specified
internal features of the backbone.
