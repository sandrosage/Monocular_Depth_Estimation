## Segmentation-guided Monocular Depth Estimation with Metric Depth for ressource-constraint devices

This repository contains code to compute depth from a single image, a directory of images and a single video. It accompanies several depth models developed by the following papers:

[Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer](https://arxiv.org/abs/1907.01341v3)

[Vision Transformers for Dense Prediction](hello.html)

[Monodepth2](https://)

For the latest release MiDaS 3.1, a [technical report](https://arxiv.org/pdf/2307.14460.pdf) and [video](https://www.youtube.com/watch?v=UjaeNNFf9sE&t=3s) are available.


MiDaS was trained on up to 12 datasets (ReDWeb, DIML, Movies, MegaDepth, WSVD, TartanAir, HRWSI, ApolloScape, BlendedMVS, IRS, KITTI, NYU Depth V2) with
multi-objective optimization. 
The original model that was trained on 5 datasets  (`MIX 5` in the paper) can be found [here](https://github.com/isl-org/MiDaS/releases/tag/v2).
The figure below shows an overview of the different MiDaS models; the bubble size scales with number of parameters.

![](figures/Improvement_vs_FPS.png)

### Setup 

1) Pick one of the estimator types and download the weights
  - Monodepth2: automatically downloads the weights in the first run
  - MiDaS and DPT: download the model from their repositories and store them inside the `weights` folder in the correct directory (e.g. `Depth_Estimation/MiDaS/weights`)
  - ZoeDepth: works with `torch.hub` and also automatically downloads the weights.

#### Github references
- Monodepth2 [^1]
[^1]: hello
- For highest quality: [dpt_beit_large_512](https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt)
- For moderately less quality, but better speed-performance trade-off: [dpt_swin2_large_384](https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_large_384.pt)
- For embedded devices: [dpt_swin2_tiny_256](https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_tiny_256.pt), [dpt_levit_224](https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_levit_224.pt)
- For inference on Intel CPUs, OpenVINO may be used for the small legacy model: openvino_midas_v21_small [.xml](https://github.com/isl-org/MiDaS/releases/download/v3_1/openvino_midas_v21_small_256.xml), [.bin](https://github.com/isl-org/MiDaS/releases/download/v3_1/openvino_midas_v21_small_256.bin)

MiDaS 3.0: Legacy transformer models [dpt_large_384](https://github.com/isl-org/MiDaS/releases/download/v3/dpt_large_384.pt) and [dpt_hybrid_384](https://github.com/isl-org/MiDaS/releases/download/v3/dpt_hybrid_384.pt)

MiDaS 2.1: Legacy convolutional models [midas_v21_384](https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_384.pt) and [midas_v21_small_256](https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt) 

1) Set up dependencies: 

Powershell:

    ```shell
    python -m venv venv
    .\venv\Scripts\activate
    pip install -r requirements.txt
    ```

#### optional

For the Next-ViT model, execute

```shell
git submodule add https://github.com/isl-org/Next-ViT midas/external/next_vit
```

For the OpenVINO model, install

```shell
pip install openvino
```
    
### Usage

1) Place one or more input images in the folder `input`.

2) Run the model with

   ```shell
   python run.py --model_type <model_type> --input_path input --output_path output
   ```
   where ```<model_type>``` is chosen from [dpt_beit_large_512](#model_type), [dpt_beit_large_384](#model_type),
   [dpt_beit_base_384](#model_type), [dpt_swin2_large_384](#model_type), [dpt_swin2_base_384](#model_type),
   [dpt_swin2_tiny_256](#model_type), [dpt_swin_large_384](#model_type), [dpt_next_vit_large_384](#model_type),
   [dpt_levit_224](#model_type), [dpt_large_384](#model_type), [dpt_hybrid_384](#model_type),
   [midas_v21_384](#model_type), [midas_v21_small_256](#model_type), [openvino_midas_v21_small_256](#model_type).
 
3) The resulting depth maps are written to the `output` folder.


#### via PyTorch Hub

The pretrained model is also available on [PyTorch Hub](https://pytorch.org/hub/intelisl_midas_v2/)

#### via TensorFlow or ONNX

See [README](https://github.com/isl-org/MiDaS/tree/master/tf) in the `tf` subdirectory.

Currently only supports MiDaS v2.1. 



### Accuracy

We provide a **zero-shot error** $\epsilon_d$ which is evaluated for 6 different datasets
(see [paper](https://arxiv.org/abs/1907.01341v3)). **Lower error values are better**. 
$\color{green}{\textsf{Overall model quality is represented by the improvement}}$ ([Imp.](#improvement)) with respect to
MiDaS 3.0 DPT<sub>L-384</sub>. The models are grouped by the height used for inference, whereas the square training resolution is given by 
the numbers in the model names. The table also shows the **number of parameters** (in millions) and the 
**frames per second** for inference at the training resolution (for GPU RTX 3090):

| MiDaS Model                                                                                                           | DIW </br><sup>WHDR</sup> | Eth3d </br><sup>AbsRel</sup> | Sintel </br><sup>AbsRel</sup> |   TUM </br><sup>δ1</sup> | KITTI </br><sup>δ1</sup> | NYUv2 </br><sup>δ1</sup> | $\color{green}{\textsf{Imp.}}$ </br><sup>%</sup> | Par.</br><sup>M</sup> | FPS</br><sup>&nbsp;</sup> |
|-----------------------------------------------------------------------------------------------------------------------|-------------------------:|-----------------------------:|------------------------------:|-------------------------:|-------------------------:|-------------------------:|-------------------------------------------------:|----------------------:|--------------------------:|
| **Inference height 512**                                                                                              |                          |                              |                               |                          |                          |                          |                                                  |                       |                           |
| [v3.1 BEiT<sub>L-512</sub>](https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt)                                                                                     |                   0.1137 |                       0.0659 |                        0.2366 |                 **6.13** |                   11.56* |                **1.86*** |                     $\color{green}{\textsf{19}}$ |               **345** |                   **5.7** |
| [v3.1 BEiT<sub>L-512</sub>](https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt)$\tiny{\square}$                                                                     |               **0.1121** |                   **0.0614** |                    **0.2090** |                     6.46 |                **5.00*** |                    1.90* |                     $\color{green}{\textsf{34}}$ |               **345** |                   **5.7** |
|                                                                                                                       |                          |                              |                               |                          |                          |                          |                                                  |                       |                           |
| **Inference height 384**                                                                                              |                          |                              |                               |                          |                          |                          |                                                  |                       |                           |
| [v3.1 BEiT<sub>L-512</sub>](https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt)                                                                                     |                   0.1245 |                       0.0681 |                    **0.2176** |                 **6.13** |                    6.28* |                **2.16*** |                     $\color{green}{\textsf{28}}$ |                   345 |                        12 |
| [v3.1 Swin2<sub>L-384</sub>](https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_large_384.pt)$\tiny{\square}$                                                                    |                   0.1106 |                       0.0732 |                        0.2442 |                     8.87 |                **5.84*** |                    2.92* |                     $\color{green}{\textsf{22}}$ |                   213 |                        41 |
| [v3.1 Swin2<sub>B-384</sub>](https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_base_384.pt)$\tiny{\square}$                                                                    |                   0.1095 |                       0.0790 |                        0.2404 |                     8.93 |                    5.97* |                    3.28* |                     $\color{green}{\textsf{22}}$ |                   102 |                        39 |
| [v3.1 Swin<sub>L-384</sub>](https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin_large_384.pt)$\tiny{\square}$                                                                     |                   0.1126 |                       0.0853 |                        0.2428 |                     8.74 |                    6.60* |                    3.34* |                     $\color{green}{\textsf{17}}$ |                   213 |                        49 |
| [v3.1 BEiT<sub>L-384</sub>](https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_384.pt)                                                                                     |                   0.1239 |                   **0.0667** |                        0.2545 |                     7.17 |                    9.84* |                    2.21* |                     $\color{green}{\textsf{17}}$ |                   344 |                        13 |
| [v3.1 Next-ViT<sub>L-384</sub>](https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_next_vit_large_384.pt)                                                                                 |               **0.1031** |                       0.0954 |                        0.2295 |                     9.21 |                    6.89* |                    3.47* |                     $\color{green}{\textsf{16}}$ |                **72** |                        30 |
| [v3.1 BEiT<sub>B-384</sub>](https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_base_384.pt)                                                                                     |                   0.1159 |                       0.0967 |                        0.2901 |                     9.88 |                   26.60* |                    3.91* |                    $\color{green}{\textsf{-31}}$ |                   112 |                        31 |
| [v3.0 DPT<sub>L-384</sub>](https://github.com/isl-org/MiDaS/releases/download/v3/dpt_large_384.pt)        |                   0.1082 |                       0.0888 |                        0.2697 |                     9.97 |                     8.46 |                     8.32 |                      $\color{green}{\textsf{0}}$ |                   344 |                    **61** |
| [v3.0 DPT<sub>H-384</sub>](https://github.com/isl-org/MiDaS/releases/download/v3/dpt_hybrid_384.pt)       |                   0.1106 |                       0.0934 |                        0.2741 |                    10.89 |                    11.56 |                     8.69 |                    $\color{green}{\textsf{-10}}$ |                   123 |                        50 |
| [v2.1 Large<sub>384</sub>](https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_384.pt)       |                   0.1295 |                       0.1155 |                        0.3285 |                    12.51 |                    16.08 |                     8.71 |                    $\color{green}{\textsf{-32}}$ |                   105 |                        47 |
|                                                                                                                       |                          |                              |                               |                          |                          |                          |                                                  |                       |                           |
| **Inference height 256**                                                                                              |                          |                              |                               |                          |                          |                          |                                                  |                       |                           |
| [v3.1 Swin2<sub>T-256</sub>](https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_tiny_256.pt)$\tiny{\square}$                                                                    |               **0.1211** |                   **0.1106** |                    **0.2868** |                **13.43** |               **10.13*** |                **5.55*** |                    $\color{green}{\textsf{-11}}$ |                    42 |                        64 |
| [v2.1 Small<sub>256</sub>](https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt) |                   0.1344 |                       0.1344 |                        0.3370 |                    14.53 |                    29.27 |                    13.43 |                    $\color{green}{\textsf{-76}}$ |                **21** |                    **90** |
|                                                                                                                       |                          |                              |                               |                          |                          |                          |                                                  |                       |                           |
| **Inference height 224**                                                                                              |                          |                              |                               |                          |                          |                          |                                                  |                       |                           |
| [v3.1 LeViT<sub>224</sub>](https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_levit_224.pt)$\tiny{\square}$                                                                      |               **0.1314** |                   **0.1206** |                    **0.3148** |                **18.21** |               **15.27*** |                **8.64*** |                    $\color{green}{\textsf{-40}}$ |                **51** |                    **73** |

&ast; No zero-shot error, because models are also trained on KITTI and NYU Depth V2\
$\square$ Validation performed at **square resolution**, either because the transformer encoder backbone of a model 
does not support non-square resolutions (Swin, Swin2, LeViT) or for comparison with these models. All other 
validations keep the aspect ratio. A difference in resolution limits the comparability of the zero-shot error and the
improvement, because these quantities are averages over the pixels of an image and do not take into account the 
advantage of more details due to a higher resolution.\
Best values per column and same validation height in bold

#### Improvement

The improvement in the above table is defined as the relative zero-shot error with respect to MiDaS v3.0 
DPT<sub>L-384</sub> and averaging over the datasets. So, if $\epsilon_d$ is the zero-shot error for dataset $d$, then
the $\color{green}{\textsf{improvement}}$ is given by $100(1-(1/6)\sum_d\epsilon_d/\epsilon_{d,\rm{DPT_{L-384}}})$%.

Note that the improvements of 10% for MiDaS v2.0 &rarr; v2.1 and 21% for MiDaS v2.1 &rarr; v3.0 are not visible from the
improvement column (Imp.) in the table but would require an evaluation with respect to MiDaS v2.1 Large<sub>384</sub>
and v2.0 Large<sub>384</sub> respectively instead of v3.0 DPT<sub>L-384</sub>.

### Depth map comparison

Zoom in for better visibility
![](figures/Comparison.png)

### Speed on Camera Feed	

Test configuration	
- Windows 10	
- 11th Gen Intel Core i7-1185G7 3.00GHz	
- 16GB RAM	
- Camera resolution 640x480	
- openvino_midas_v21_small_256	

Speed: 22 FPS

### Applications

MiDaS is used in the following other projects from Intel Labs:

- [ZoeDepth](https://arxiv.org/pdf/2302.12288.pdf) (code available [here](https://github.com/isl-org/ZoeDepth)): MiDaS computes the relative depth map given an image. For metric depth estimation, ZoeDepth can be used, which combines MiDaS with a metric depth binning module appended to the decoder.
- [LDM3D](https://arxiv.org/pdf/2305.10853.pdf) (Hugging Face model available [here](https://huggingface.co/Intel/ldm3d-4c)): LDM3D is an extension of vanilla stable diffusion designed to generate joint image and depth data from a text prompt. The depth maps used for supervision when training LDM3D have been computed using MiDaS.

### Changelog

* [Dec 2022] Released [MiDaS v3.1](https://arxiv.org/pdf/2307.14460.pdf):
    - New models based on 5 different types of transformers ([BEiT](https://arxiv.org/pdf/2106.08254.pdf), [Swin2](https://arxiv.org/pdf/2111.09883.pdf), [Swin](https://arxiv.org/pdf/2103.14030.pdf), [Next-ViT](https://arxiv.org/pdf/2207.05501.pdf), [LeViT](https://arxiv.org/pdf/2104.01136.pdf))
    - Training datasets extended from 10 to 12, including also KITTI and NYU Depth V2 using [BTS](https://github.com/cleinc/bts) split
    - Best model, BEiT<sub>Large 512</sub>, with resolution 512x512, is on average about [28% more accurate](#Accuracy) than MiDaS v3.0
    - Integrated live depth estimation from camera feed
* [Sep 2021] Integrated to [Huggingface Spaces](https://huggingface.co/spaces) with [Gradio](https://github.com/gradio-app/gradio). See [Gradio Web Demo](https://huggingface.co/spaces/akhaliq/DPT-Large).
* [Apr 2021] Released MiDaS v3.0:
    - New models based on [Dense Prediction Transformers](https://arxiv.org/abs/2103.13413) are on average [21% more accurate](#Accuracy) than MiDaS v2.1
    - Additional models can be found [here](https://github.com/isl-org/DPT)
* [Nov 2020] Released MiDaS v2.1:
	- New model that was trained on 10 datasets and is on average about [10% more accurate](#Accuracy) than [MiDaS v2.0](https://github.com/isl-org/MiDaS/releases/tag/v2)
	- New light-weight model that achieves [real-time performance](https://github.com/isl-org/MiDaS/tree/master/mobile) on mobile platforms.
	- Sample applications for [iOS](https://github.com/isl-org/MiDaS/tree/master/mobile/ios) and [Android](https://github.com/isl-org/MiDaS/tree/master/mobile/android)
	- [ROS package](https://github.com/isl-org/MiDaS/tree/master/ros) for easy deployment on robots
* [Jul 2020] Added TensorFlow and ONNX code. Added [online demo](http://35.202.76.57/).
* [Dec 2019] Released new version of MiDaS - the new model is significantly more accurate and robust
* [Jul 2019] Initial release of MiDaS ([Link](https://github.com/isl-org/MiDaS/releases/tag/v1))

### Citation

Please cite our paper if you use this code or any of the models:
```
@ARTICLE {Ranftl2022,
    author  = "Ren\'{e} Ranftl and Katrin Lasinger and David Hafner and Konrad Schindler and Vladlen Koltun",
    title   = "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer",
    journal = "IEEE Transactions on Pattern Analysis and Machine Intelligence",
    year    = "2022",
    volume  = "44",
    number  = "3"
}
```

If you use a DPT-based model, please also cite:

```
@article{Ranftl2021,
	author    = {Ren\'{e} Ranftl and Alexey Bochkovskiy and Vladlen Koltun},
	title     = {Vision Transformers for Dense Prediction},
	journal   = {ICCV},
	year      = {2021},
}
```

Please cite the technical report for MiDaS 3.1 models:

```
@article{birkl2023midas,
      title={MiDaS v3.1 -- A Model Zoo for Robust Monocular Relative Depth Estimation},
      author={Reiner Birkl and Diana Wofk and Matthias M{\"u}ller},
      journal={arXiv preprint arXiv:2307.14460},
      year={2023}
}
```

For ZoeDepth, please use

```
@article{bhat2023zoedepth,
  title={Zoedepth: Zero-shot transfer by combining relative and metric depth},
  author={Bhat, Shariq Farooq and Birkl, Reiner and Wofk, Diana and Wonka, Peter and M{\"u}ller, Matthias},
  journal={arXiv preprint arXiv:2302.12288},
  year={2023}
}
```

and for LDM3D

```
@article{stan2023ldm3d,
  title={LDM3D: Latent Diffusion Model for 3D},
  author={Stan, Gabriela Ben Melech and Wofk, Diana and Fox, Scottie and Redden, Alex and Saxton, Will and Yu, Jean and Aflalo, Estelle and Tseng, Shao-Yen and Nonato, Fabio and Muller, Matthias and others},
  journal={arXiv preprint arXiv:2305.10853},
  year={2023}
}
```

### Acknowledgements

Our work builds on and uses code from [timm](https://github.com/rwightman/pytorch-image-models) and [Next-ViT](https://github.com/bytedance/Next-ViT). 
We'd like to thank the authors for making these libraries available.

### License 

MIT License 
