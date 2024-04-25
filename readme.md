## Segmentation-guided Monocular Depth Estimation with Metric Depth for ressource-constraint devices

This repository contains code to compute depth from a single image, a directory of images and a single video. It accompanies several depth models developed by the following papers:

[Digging Into Self-Supervised Monocular Depth Estimation](https://arxiv.org/pdf/1806.01260.pdf)

[Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer](https://arxiv.org/pdf/1907.01341v3.pdf)

[MiDaS v3.1 – A Model Zoo for Robust Monocular Relative Depth Estimation](https://arxiv.org/pdf/2307.14460.pdf)

[Vision Transformers for Dense Prediction](https://arxiv.org/pdf/2103.13413.pdf)

[ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth](https://arxiv.org/pdf/2302.12288.pdf)

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

### Setup 

1) Pick one of the estimator types and download the weights
  - Monodepth2: automatically downloads the weights in the first run
  - MiDaS and DPT: download the model from their repositories and store them inside the `weights` folder in the correct directory (e.g. `Depth_Estimation/MiDaS/weights`)
  - ZoeDepth: works with `torch.hub` and also automatically downloads the weights.


2) Set up dependencies: 

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
   python run.py -et <model_type> -s
   ````
   where `<estimator_type>` is chosen from `[Mono2, MiDaS, DPT, ZoeDepth]`. If the `-s` flag is set, the image is also segmentated.
 
3) The resulting depth maps are written to the `output/images` folder. For each of the estimator types there exists a subdirectory where all the depth maps are stored. The depth maps are accordingly named by their input image name and their specific model type. Additionally the segmented image and the mean metric depth per segmented object is stored in a `*_mean_depth_per_object.csv`. This file contains the class name and the mean metric depth in asceding order of the depth.

### License 

MIT License 
