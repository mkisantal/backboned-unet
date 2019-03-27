from setuptools import setup

setup(name='backboned_unet',
      version='0.0.1',
      description='U-Net built with TorchVision backbones.',
      url='https://github.com/mkisantal/backboned-unet',
      keywords='machine deep learning neural networks pytorch torchvision segmentation unet',
      author='mate Kisantal',
      author_email='kisantal.mate@gmail.com',
      license='MIT',
      packages=['backboned_unet'],
      install_requires=[
          'torch',
          'torchvision'
      ],
      zip_safe=False)
