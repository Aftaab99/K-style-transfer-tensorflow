# K-Style Transfer for Mobile

A tensorflow implementation of a [Learned Representation of artistic style](https://arxiv.org/abs/1610.07629)

Most of the code in the repo was taken from [hwalsuklee/tensorflow-fast-style-transfer](https://github.com/hwalsuklee/tensorflow-fast-style-transfer). 

This is the implementation used in my app [Spectrum - Artistic Photo editor](https://play.google.com/store/apps/details?id=com.spectrumeditor.aftaab.spectrumJ).  If this fork helps you, please consider downloading the app, it really helps me out.

### Changes/Improvements
1. Added Conditional Instance Normalization which enables a single model to learn multiple styles with only a few extra parameters.
2. Used Upsampling+Convolution instead of Transpose Convolutions([prevents checkerboard artifacts](https://distill.pub/2016/deconv-checkerboard/)).
3. Experimented with Depthwise separable convolutions instead of regular convolutions. The current model still uses all convolutional layers but you can replace `_conv_layer` with `_depthwise_conv_layer` for some layers to reduce the model size(but don't do this for the initial layers).

### Training
I trained my models in this [Kaggle kernel](https://www.kaggle.com/aftaab/styletransferondevice/). Please see `run_train.py` and `run_test.py` for training options.

### Results
[Spectrum Landing Page](https://aftaab99.github.io/spectrum/)
