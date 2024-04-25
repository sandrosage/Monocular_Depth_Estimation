## Segmentation-guided Monocular Depth Estimation with Metric Depth for ressource-constraint devices

This repository contains code to compute depth from a single image, a directory of images and a single video. It accompanies several depth models developed by the following papers:

[Digging Into Self-Supervised Monocular Depth Estimation](https://arxiv.org/pdf/1806.01260.pdf)

[Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer](https://arxiv.org/pdf/1907.01341v3.pdf)

[MiDaS v3.1 – A Model Zoo for Robust Monocular Relative Depth Estimation](https://arxiv.org/pdf/2307.14460.pdf)

[Vision Transformers for Dense Prediction](https://arxiv.org/pdf/2103.13413.pdf)

[ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth](https://arxiv.org/pdf/2302.12288.pdf)

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

For further informations and how to work with the models itself, refer to the original github repositories below.

- Monodepth2 [^1]
- MiDaS [^2]
- DPT [^3]
- ZoeDepth [^4]
[^1]: https://github.com/nianticlabs/monodepth2
[^2]: https://github.com/isl-org/MiDaS
[^3]: https://github.com/isl-org/DPT
[^4]: https://github.com/isl-org/ZoeDepth


1) Set up dependencies: 

Powershell:

    ```shell
    python -m venv venv
    .\venv\Scripts\activate
    pip install -r requirements.txt
    ```
    
### Usage

1) Place one or more input images in the folder `input/images`.

2) Run the model with

   ```shell
   python run.py --model_type <model_type> --input_path input --output_path output
   ```
   where ```<estimator_type>``` is chosen from [dpt_beit_large_512](#model_type), [dpt_beit_large_384](#model_type),
   [dpt_beit_base_384](#model_type), [dpt_swin2_large_384](#model_type), [dpt_swin2_base_384](#model_type),
   [dpt_swin2_tiny_256](#model_type), [dpt_swin_large_384](#model_type), [dpt_next_vit_large_384](#model_type),
   [dpt_levit_224](#model_type), [dpt_large_384](#model_type), [dpt_hybrid_384](#model_type),
   [midas_v21_384](#model_type), [midas_v21_small_256](#model_type), [openvino_midas_v21_small_256](#model_type).
 
3) The resulting depth maps are written to the `output/images` folder. For each of the estimator types there exists a subdirectory where all the depth maps are stored. The depth maps are accordingly named by their input image name and their specific model type.

### License 

MIT License 
