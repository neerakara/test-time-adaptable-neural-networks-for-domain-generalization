# domain_generalization_image_segmentation

Code for the paper "Test-time adaptable neural networks for robust medical image segmentation": https://arxiv.org/abs/2004.04668

The method consists of three steps:
1. Train a segmentation network on the source domain: train_i2l_mapper.py
2. Train a denoising autoencoder on the source domain labels: train_l2l_mapper.py
3. Adapt the normalization module of the segmentation network for each test image: update_i2i_mapper.py
