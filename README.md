# denoising-conv-autoencoder

Denoising convolutional autoencoder for the MNIST handwritten digit database.

* Salt, peper, and random noise is added to the MNIST handwritten digit database.
* Parameters allow to control the amount of noise
* The new dataset can be store so it is not required to compute/add the noise every time
* Everything is moved to the GPU (since the dataset is not so big)

Digits with noise (salt and peper, and random noise): 
![](figures/noisy.png)
Denoised digits using denoising convolutional neural network: 
![](figures/denoised.png)
GIF for comparison: 
![](figures/comparison.gif)
