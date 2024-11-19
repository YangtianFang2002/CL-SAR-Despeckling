# Contrastive Learning for SAR Despeckling
### 📖[**Paper**](https://www.sciencedirect.com/science/article/pii/S0924271624004118)

PyTorch codes for "[Contrastive Learning for SAR Despeckling](https://www.sciencedirect.com/science/article/pii/S0924271624004118)", **ISPRS Journal of Photogrammetry and Remote Sensing**, 2024.

### Abstract
>The use of synthetic aperture radar (SAR) has greatly improved our ability to capture high-resolution terrestrial images under various weather conditions. However, SAR imagery is affected by speckle noise, which distorts image details and hampers subsequent applications. Recent forays into supervised deep learning-based denoising methods, like MRDDANet and SAR-CAM, offer a promising avenue for SAR despeckling. However, they are impeded by the domain gaps between synthetic data and realistic SAR images. 
To tackle this problem, we introduce a self-supervised speckle-aware network to utilize the limited near-real datasets and unlimited synthetic datasets simultaneously, which boosts the performance of the downstream despeckling module by teaching the module to discriminate the domain gap of different datasets in the embedding space. Specifically, based on contrastive learning, the speckle-aware network first characterizes the discriminative representations of spatial-correlated speckle noise in different images across diverse datasets, which provides priors of versatile speckles and image characteristics. Then, the representations are effectively modulated into a subsequent multi-scale despeckling network to generate authentic despeckled images. In this way, the despeckling module can reconstruct reliable SAR image characteristics by learning from near-real datasets, while the generalization performance is guaranteed by learning abundant patterns from synthetic datasets simultaneously. Additionally, a novel excitation aggregation pooling module is inserted into the despeckling network to enhance the network further, which utilizes features from different levels of scales and better preserves spatial details around strong scatters in real SAR images. Extensive experiments across real SAR datasets from Sentinel-1, Capella-X, and TerraSAR-X satellites are carried out to verify the effectiveness of the proposed method over other state-of-the-art methods. Specifically, the proposed method achieves the best PSNR and SSIM values evaluated on the near-real Sentinel-1 dataset, with gains of 0.22dB in PSNR compared to MRDDANet, and improvements of 1.3\% in SSIM over SAR-CAM. The code will be available at https://github.com/YangtianFang2002/CL-SAR-Despeckling.
>

### Network
 ![image](/img/structure.jpg)

## 🧩Install
```
git clone https://github.com/YangtianFang2002/CL-SAR-Despeckling.git
```
### Requirements
> - Python 3.11.5
> - PyTorch >= 2.4.0
> - Ubuntu 22.04.4 LTS, cuda-12.1

### Dataset Preparation (Offline)
Please download the datasets from github `release` and unzip to folder `datasets`.

The `datasets` folder should contain `AID`, `SARdata` and multiple `txt` files as training labels.

## Usage

### Eval
given the checkpoint at `experiments/MDN1-default/models/net_g_latest.pth`, run `eval.sh`
sample output:
```
Using experiments/MDN1-default/models/net_g_latest.pth
python basicsr/train.py -opt options/real/MDN1-default.yml --val --force_yml datasets:val:name=ValSetAll datasets:val:meta_info=datasets/val.txt path:pretrain_network_g=experiments/MDN1-default/models/net_g_latest.pth &> experiments/MDN1-default/models/net_g_latest.pth.txt
2024-10-31 17:29:42,045 INFO: Validation ValSetAll,              # psnr: 40.088746113375         # ssim: 0.952861683206  # MOR: 1.359815579653   # CVOR: 1.999100595713      # SNR: 49.955084800720  # MSE: 0.000195074164   # ssim_view: 0.830520230517     # psnr_view: 25.340706002133
python basicsr/train.py -opt options/real/MDN1-default.yml --val --force_yml datasets:val:name=ValSetMountains datasets:val:meta_info=datasets/val_mountains.txt path:pretrain_network_g=experiments/MDN1-default/models/net_g_latest.pth &> experiments/MDN1-default/models/net_g_latest.pth.txt
2024-10-31 17:29:52,647 INFO: Validation ValSetMountains,                # psnr: 35.206861697798         # ssim: 0.913796763046  # MOR: 1.489311456680   # CVOR: 3.211423601423      # SNR: 47.070822034563  # MSE: 0.000390892518   # ssim_view: 0.816974386488     # psnr_view: 25.724229357030
python basicsr/train.py -opt options/real/MDN1-default.yml --val --force_yml datasets:val:name=ValSetHills datasets:val:meta_info=datasets/val_hills.txt path:pretrain_network_g=experiments/MDN1-default/models/net_g_latest.pth &> experiments/MDN1-default/models/net_g_latest.pth.txt
2024-10-31 17:30:03,039 INFO: Validation ValSetHills,            # psnr: 40.482942332329         # ssim: 0.960478330739  # MOR: 1.277799288432   # CVOR: 1.037530908982      # SNR: 48.789754708608  # MSE: 0.000108912639   # ssim_view: 0.845169412011     # psnr_view: 25.695244021084
python basicsr/train.py -opt options/real/MDN1-default.yml --val --force_yml datasets:val:name=ValSetPlains datasets:val:meta_info=datasets/val_plains.txt path:pretrain_network_g=experiments/MDN1-default/models/net_g_latest.pth &> experiments/MDN1-default/models/net_g_latest.pth.txt
2024-10-31 17:30:13,992 INFO: Validation ValSetPlains,           # psnr: 44.632747447624         # ssim: 0.985398046986  # MOR: 1.300619465964   # CVOR: 1.610980638436      # SNR: 53.838201931545  # MSE: 0.000073108744   # ssim_view: 0.831509640581     # psnr_view: 24.653291262111
```
where metric `ssim_view` and `psnr_view` correspond to the results of the evaluation on the real SAR images of Sentinel-1.

The visualization result can be reviewed at `experiments/MDN1-default/visualization`, where subfolder `eval_I` contains the numpy serialization file `.npy` for lossless model despeckled output in intensity.

### Test
given the checkpoint at `experiments/MDN1-default/models/net_g_latest.pth`, run `test.sh`
sample output:
```
Using experiments/MDN1-default/models/net_g_latest.pth
python basicsr/predict.py -i datasets/SARdata/multitest/Capella-X -opt options/real/MDN1-default.yml --force_yml path:pretrain_network_g=experiments/MDN1-default/models/net_g_latest.pth &> experiments/MDN1-default/models/net_g_latest.pth.txt
2024-10-31 17:33:20,698 INFO: Tested 2 images from Capella-X, Final Metric: # SNR: 37.3225 
Tested 5_20201228.rmli  SNR: 37.3297
Tested 7_20210923.rmli  SNR: 37.3152
2024-10-31 17:33:20,698 INFO: Tested 2 images from Capella-X, Final Metric: # SNR: 37.3225 
python basicsr/predict.py -i datasets/SARdata/multitest/TerraSAR&Tandem-X -opt options/real/MDN1-default.yml --force_yml path:pretrain_network_g=experiments/MDN1-default/models/net_g_latest.pth &> experiments/MDN1-default/models/net_g_latest.pth.txt
2024-10-31 17:33:27,057 INFO: Tested 2 images from TerraSAR&Tandem-X, Final Metric: # SNR: 44.1173 
Tested 9_TDX1_20120805.rmli     SNR: 39.5712
Tested 4_TDX1_20111126.rmli     SNR: 48.6635
2024-10-31 17:33:27,057 INFO: Tested 2 images from TerraSAR&Tandem-X, Final Metric: # SNR: 44.1173 
```
The visualization result can be reviewed at `experiments/MDN1-default/visualization/test`.

### Train
modify `options/real/MDN1-t1.yml` and run:
```
python basicsr/train.py -opt options/real/MDN1-t1.yml
```

## Acknowledgement
Our work mainly borrows from [DRSR](https://github.com/XY-boy/DRSR) and [SimCLR](https://github.com/sthalles/SimCLR). Thanks to these excellent works!

## Contact
If you have any questions or suggestions, feel free to contact me. 😊  
Email: justin62628@whu.edu.cn;

## Citation
If you find our work helpful in your research, kindly consider citing it. We appreciate your support！😊

```
@article{FANG2024376,
title = {Contrastive learning for real SAR image despeckling},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {218},
pages = {376-391},
year = {2024},
issn = {0924-2716},
doi = {https://doi.org/10.1016/j.isprsjprs.2024.11.003},
url = {https://www.sciencedirect.com/science/article/pii/S0924271624004118},
author = {Yangtian Fang and Rui Liu and Yini Peng and Jianjun Guan and Duidui Li and Xin Tian},
keywords = {Real SAR despeckling, Self-supervised learning, Contrastive learning, Multi-scale despeckling network, Excitation aggregation pooling},
}

```
