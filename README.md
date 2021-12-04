## [Progressive Image Deraining Networks: A Better and Simpler Baseline]

### Introduction
In recent years, single image deraining has received considerable research interests.
Supervised learning is widely adopted for training dedicated deraining networks to achieve promising results on synthetic datasets, while limiting in handling real-world rainy images. 
Unsupervised and semi-supervised learning-based deranining methods have been studied to improve the performance on real cases, but their quantitative results are still inferior.
In this paper, we propose to address this crucial issue for image deraining in terms of backbone architecture and the strategy of semi-supervised learning. 
First, in terms of network architecture, we propose an attentive image deraining network (**AIDNet**), where residual attention block is proposed to exploit the beneficial deep feature from the rain streak layer to background image layer.
Then, different from the traditional semi-supervised method by enforcing the consistency of rain pattern distribution between real rainy images and synthetic rainy images, we explore the correlation between the real clean images and the precdicted background image by imposing adversarial losses in wavelet space (**I_HH**), (**I_HL**), and (**I_LH**), resulting in the final (**AID-DWT**) model.
Extensive experiments on both synthetic and real-world rainy images have validated that our (**AID-DWT**) can achieve better deraining results than not only existing semi-supervised deraining methods qualitatively but also outperform state-of-the-art supervised deraining methods quantitatively.

## Prerequisites
- Python 3.6, PyTorch >= 1.0.*
- Platforms: Ubuntu 16.04, cuda-10.0 & cuDNN v-7.3 (higher versions also work well)


## Datasets

PRN and PReNet are evaluated on four datasets*: 
Rain200H [1], Rain1200 [2], Rain1400 [3] Rain12 [4] and SPA [5]. 
 

## Getting Started

### 1) Testing

We have placed our pre-trained models into `./logs/DWT/`. 

Run shell scripts to test the models:
```bash
python test_cbam_dwt_Rain12.py   # test models on Rain12
bash test_cbam_dwt_Rain1200.py   # test models on Rain1200
bash test_cbam_dwt_Rain1400.py   # test models on Rain1400
bash test_cbam_dwt_Rain200H.py   # test models on Rain200H 
bash test_cbam_dwt_SPA.py        # test models on SPA
```

###
Average PSNR/SSIM values on four datasets:

Dataset      |NLEDN[6]     |ReHEN[7]     |PReNet[8]    |RPDNet[9]    |MSPFN[10]    |Syn2Real[11] |AID-DWT     
-------------|-------------|-------------|-------------|-------------|-------------|-------------
Rain200H     |27.315/0.8904|27.525/0.8663|27.883/0.8908|27.909/0.8923|25.554/0.8039|22.825/0.7114|28.903/0.9074
Rain1200     |30.799/0.9127|30.456/0.8702|27.307/0.8712|26.486/0.8401|30.390/0.8862|28.386/0.8275|31.960/0.9136
Rain1400     |30.808/0.9181|30.984/0.9156|30.609/0.9181|30.772/0.9178|30.016/0.9164|28.360/0.8574|31.001/0.9246
Rain12       |33.028/0.9615|35.095/0.9400|34.791/0.9644|35.055/0.9657|34.253/0.9469|25.199/0.8497|35.587/0.9679
SPA          |30.596/0.9363|32.652/0.9297|32.720/0.9317|32.803/0.9337|29.538/0.9193|31.824/0.9307|33.263/0.9375

###
Average NIQE values on real datasets:

Dataset      |NLEDN[6]     |ReHEN[7]     |PReNet[8]    |RPDNet[9]    |MSPFN[10]    |Syn2Real[11] |AID-DWT     
-------------|-------------|-------------|-------------|-------------|-------------|-------------
Real275      |3.5554       |3.7355       |3.7745       |3.8957       |3.8616       |4.0372       |3.5519

### 3) Training

Run shell scripts to train the models:
```bash
python train_cbam_dwt_Rain1200.py   # train models on Rain1200  
python train_cbam_dwt_Rain1400.py   # train models on Rain1400
python train_cbam_dwt_Rain200H.py   # train models on Rain200H     
``` 

### Model Configuration

The following tables provide the configurations of options. 

#### Training Mode Configurations

Option                 |Default        | Description
-----------------------|---------------|------------
batchSize              | 18            | Training batch size
recurrent_iter         | 8             | Number of recursive stages
epochs                 | 100           | Number of training epochs
milestone              | [30,50,80]    | When to decay learning rate
lr                     | 1e-3          | Initial learning rate
save_freq              | 1             | save intermediate model
use_GPU                | True          | use GPU or not
gpu_id                 | 0,1           | GPU id
data_path              | N/A           | path to training images
save_path              | N/A           | path to save models and status           

#### Testing Mode Configurations

Option                 |Default           | Description
-----------------------|------------------|------------
use_GPU                | True             | use GPU or not
gpu_id                 | 0,1                | GPU id
recurrent_iter         | 8                | Number of recursive stages
logdir                 | N/A              | path to trained model
data_path              | N/A              | path to testing images
save_path              | N/A              | path to save results

## References
[1] W. Yang, R.T. Feng, J.L.Z.G., Yan, S. Deep joint rain detection and removal from a single image. In IEEE CVPR 2017.

[2] Zhang, H., Patel, V.M. Density-aware single image de-raining using a multi-stream dense network. In IEEE CVPR 2018.

[3] X. Fu, J. Huang, D.Z.Y.H.X.D., Paisley, J. Removing rain from single images via a deep detail network. In IEEE CVPR 2017.

[4] Li, Y., Tan, R.T., Guo, X., Lu, J., Brown, M.S. Rain streak removal using layer priors. In IEEE CVPR 2016.

[5] T. Wang, X. Yang, K.X.S.C.Q.Z., Lau, R. Spatial attentive single-image deraining with a high quality real rain dataset. In IEEE CVPR 2019.

[6] Li, G., He, X., Zhang, W., Chang, H., Dong, L., Lin, L. Non-locally enhanced encoder-decoder network for single image de-raining. In ACM MM 2018.

[7] Yang, Y., Lu, H. Single image deraining via recurrent hierarchy enhancement network. In ACM MM 2019.

[8] Ren, D., Zuo, W., Hu, Q., Zhu, P., Meng, D. Progressive image deraining networks: A better and simpler baseline. In IEEE CVPR 2019.

[9] Pang, B., Zhai, D., Jiang, J., Liu, X. Single image deraining via scale-space invariant attention neural network. In ACM MM 2020.

[10] Jiang, K., Wang, Z., Yi, P., Chen, C., Huang, B., Luo, Y., Ma, J., Jiang, J. Multiscale progressive fusion network for single image deraining. In IEEE CVPR 2020.

[11] Yasarla, R., Sindagi, V.A., Patel, V.M. Syn2real transfer learning for image deraining using gaussian processes. In IEEE CVPR 2020.


# Citation

```
 @inproceedings{ren2019progressive,
   title={Semi-supervised Single Image Deraining with Discrete Wavelet Transform},
   author={Xin Cui, Wei Shang, Dongwei Ren, Pengfei Zhu, Yankun Gao},
   booktitle={Pacific Rim International Conference on Artificial Intelligence},
   year={2021},
 }
 ```
