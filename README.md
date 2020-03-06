# Keras Attention Augmented Convolutions

A Keras (Tensorflow only) wrapper over the Attention Augmentation module from the paper [Attention Augmented Convolutional Networks](https://arxiv.org/abs/1904.09925).

Provides a Layer for Attention Augmentation as well as a callable function to build a augmented convolution block.

<img src="https://github.com/titu1994/keras-attention-augmented-convs/blob/master/images/attention-augmented-convs.PNG?raw=true" height=100% width=100%>

# Usage

It is advisable to use the `augmented_conv2d(...)` function directly to build an attention augmented convolution block.

```python
from attn_augconv import augmented_conv2d

ip = Input(...)
x = augmented_conv2d(ip, ...)
...
```

If you wish to add the attention module seperately, you can do so using the `AttentionAugmentation1D` layer as well.

```python
from attn_augconv import AttentionAugmentation1D

ip = Input(...)

# make sure that input to the AttentionAugmentation1D layer has (2 * depth_k + depth_v) filters.
x = Conv2D(2 * depth_k + depth_v, ...)(ip)
x = AttentionAugmentation1D(depth_k, depth_v, num_heads)(x)
...
```

# Requirements
  - Tensorflow 2.0+
