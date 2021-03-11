# denoising-conv-autoencoder

Denoising convolutional autoencoder for the MNIST handwritten digit database.

* Salt, peper, and random noise is added to the MNIST handwritten digit database.
* Parameters allow to control the amount of noise
* The new dataset can be store so it is not required to compute/add the noise every time
* Everything is moved to the GPU (since the dataset is not so big)
    * [Download](https://drive.google.com/file/d/16AUgKIYShhEpBGs2WGihy-P6veVG8rJU/view?usp=sharing) the noisy dataset and extract it inside project's folder      


| Dataset       | 25%                     | 50%                   | [10%, 75%]            |
| :-----:       | :-----:                 |:-----:                |:-----:                |
| Original      | ![](figures/ori1.png)   | ![](figures/ori2.png) | ![](figures/ori3.png) |
| Noisy         | ![](figures/noi1.png)   | ![](figures/noi2.png) | ![](figures/noi3.png) |
| [10%, 75%]    | ![](figures/dec1.png)   | ![](figures/dec2.png) | ![](figures/dec3.png) |
