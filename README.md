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
