## Pretrained models
The networks in `networks_tf.py` use TensorFlow-compatibility functions (padding, down-sampling), while the networks in `networks.py` do not. In order to adjust the weights to the different settings, the model was trained on Places2/CelebA-HQ for some time using the pretrained weights as initialization.

Download fine-tuned weights: [CelebA-HQ](https://drive.google.com/u/0/uc?id=17oJ1dJ9O3hkl2pnl8l2PtNVf2WhSDtB7&export=download) (for `networks.py`)
