## SMMetrDE: Segmentation-guided Monocular Metric Depth Estimation

This repository contains the implementation of processing an image or a directory of images for segmentation and metric depth estimation. It accompanies several depth models developed by the following papers:

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
1) Clone the main repository and its submodules:

  ```shell
  git submodule update --init --recursive
  ```
2) Pick one of the estimator types and download the weights
  - Monodepth2: automatically downloads the weights in the first run
  - MiDaS and DPT: download the model from their repositories and store them inside the `weights` folder in the correct directory (e.g. `Depth_Estimation/MiDaS/weights`) 

    → DPT: [dpt_hybrid](https://github.com/sandrosage/Monocular_Depth_Estimation/releases/download/initial/dpt_hybrid-midas-501f0c75.pt), [dpt_hybrid_kitti](https://github.com/sandrosage/Monocular_Depth_Estimation/releases/download/initial/dpt_hybrid_kitti-cb926ef4.pt)
    
    → MiDaS: [dpt_swin2_tiny_256](https://github.com/sandrosage/Monocular_Depth_Estimation/releases/download/initial/dpt_swin2_tiny_256.pt), [midas_v21_384](https://github.com/sandrosage/Monocular_Depth_Estimation/releases/download/initial/midas_v21_384.pt), [dpt_beil_large_384](https://github.com/sandrosage/Monocular_Depth_Estimation/releases/download/initial/dpt_beit_large_384.pt)
  - ZoeDepth: works with `torch.hub` and also automatically downloads the weights.

3) Download the segmentator model weights from ([here](https://github.com/sandrosage/Monocular_Depth_Estimation/releases/download/initial/DeepLabV3Plus_resnet50.onnx))

4) Download ([here](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_selection.zip)) the KITTI Depth Prediction Evaluation [^5] validation and test sets into root directory (needed for alignment and some images can be later also moved to the `input/images`) 

[^5]: https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction

4) Set up dependencies: 

    Powershell:

      ```
      python -m venv venv
      .\venv\Scripts\activate
      pip install -r requirements.txt
      ```
    
### Usage

1) Place one or more input images in the folder `input/images`.

2) Run the model with

   ```shell
   python run.py -et <estimator_type> -s
   ````
   where `<estimator_type>` is chosen from `[Mono2, MiDaS, DPT, ZoeDepth]`. If the `-s` flag is set, the image is also segmentated.

    Additionally also choose one of the model types from the according model type list within the `run.py`.
 
3) The resulting depth maps are written to the `output/images` folder. For each of the estimator types there exists a subdirectory where all the depth maps are stored. The depth maps are accordingly named by their input image name and their specific model type. Additionally the segmented image and the mean metric depth per segmented object is stored in a `*_mean_depth_per_object.csv`. This file contains the class name and the mean metric depth in asceding order of the depth.

4) For MiDaS and DPT alignments have to be done in the first run (relative depth - metric depth) and the steps are stored in the `assets` folder.

### Evaluation
For evaluating the different depth estimators, you have to make sure that the KITTI Depth Prediction Evaluation dataset is downloaded. You just have to specify the name to the raw predictions of the model and the path to the ground truth depth maps. One example to do this for the Mono2 is the following:
```shell
python .\evaluation.py --pred_path ".\output\images\Mono2\mono+stereo_640x192\raw" --gt_path ".\depth_selection\val_selection_cropped\groundtruth_depth\" --garg_crop
```
### 👩‍⚖️ License
Copyright © Sandro Sage.
All rights reserved.
Please see the [license file](LICENSE) for terms.
