# Train MatAnyone

## Datasets

### Matting Datasets

* VM800
  * We collected 826 foreground green-screen videos from stock footage websites (Storyblocks, Envato Elements, and Motion Array).
  * These videos were processed using Adobe After Effects to extract high-quality foregrounds and alpha mattes. Please refer to **Section I (Dataset)** in our [paper](https://arxiv.org/abs/2501.14677) for details of processing pipeline.
  * Due to licensing restrictions, the VM800 dataset **cannot be publicly released**.

* Alternative Dataset
  * As an alternative, you may train the model using **[VideoMatte240K](https://grail.cs.washington.edu/projects/background-matting-v2/#/datasets)**, which provides 475 foreground videos with corresponding alpha mattes.

* ImageMatte
    * **We folllow the practice of [RVM](https://github.com/PeterL1n/RobustVideoMatting/blob/master/documentation/training.md) to collect ImageMatte.**
    * ImageMatte consists of [Distinctions-646](https://wukaoliu.github.io/HAttMatting/) and [Adobe Image Matting](https://sites.google.com/view/deepimagematting) datasets.
    * You need to contact their authors to acquire.
    * After downloading both datasets, merge their samples together to form ImageMatte dataset.
    * Only keep samples of humans.
    * Full list of images we used in ImageMatte for training:
        * [imagematte_train.txt](/documentation/misc/imagematte_train.txt)
        * [imagematte_valid.txt](/documentation/misc/imagematte_valid.txt)
    * Full list of images we used for evaluation.
        * [aim_test.txt](/documentation/misc/aim_test.txt)
        * [d646_test.txt](/documentation/misc/d646_test.txt)


### Background Datasets
* Video Backgrounds
    * We folllow the practice of [RVM](https://github.com/PeterL1n/RobustVideoMatting/blob/master/documentation/training.md) to collect Video Backgrounds.
    * We process from [DVM Background Set](https://github.com/nowsyn/DVM) by selecting clips without humans and extract only the first 100 frames as JPEG sequence.
    * Full list of clips we used:
        * [dvm_background_train_clips.txt](/documentation/misc/dvm_background_train_clips.txt)
        * [dvm_background_test_clips.txt](/documentation/misc/dvm_background_test_clips.txt)
    * You can download our preprocessed versions:
        * [Train set (14.6G)](https://robustvideomatting.blob.core.windows.net/data/BackgroundVideosTrain.tar) (Manually move some clips to validation set)
        * [Test set (936M)](https://robustvideomatting.blob.core.windows.net/data/BackgroundVideosTest.tar) (Not needed for training. Only used for making synthetic test samples for evaluation)
* Image Backgrounds
    * We download from [BG-20k](https://github.com/JizhiziLi/GFM?tab=readme-ov-file#bg-20k), which contains 20,000 high-resolution background images excluded salient objects

### Segmentation Datasets

* **We folllow the practice of [RVM](https://github.com/PeterL1n/RobustVideoMatting/blob/master/documentation/training.md) to collect Segmentation Datasets.**

* [COCO](https://cocodataset.org/#download)
    * Download [train2017.zip (18G)](http://images.cocodataset.org/zips/train2017.zip)
    * Download [panoptic_annotations_trainval2017.zip (821M)](http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip)
    * Note that our train script expects the panopitc version.
* [YouTubeVIS 2021](https://youtube-vos.org/dataset/vis/)
    * Download the train set. No preprocessing needed.
* [Supervisely Person Dataset](https://supervise.ly/explore/projects/supervisely-person-dataset-23304/datasets)
    * We used the supervisedly library to convert their encoding to bitmaps masks before using our script. We also resized down some of the large images to avoid disk loading bottleneck.
    * You can refer to [spd_preprocess.py](/documentation/misc/spd_preprocess.py)
    * Or, you can download our [preprocessed version (800M)](https://robustvideomatting.blob.core.windows.net/data/SuperviselyPersonDataset.tar)

-----

After you have downloaded the datasets. Please configure `matanyone/config/data/datasets.yaml` to provide paths to your datasets. To run the training scripts, the directory structure should look like this under your data folder:

```bash
├── mat_vid
│   ├── VM800 (VideoMatte240K as alternative)
│   │   ├── fgr
│   │   └── pha
│   ├── BG20k
│   │   └── train
│   └── DVM
│       └── train
├── mat_img
│   └── ImageMatte
│       └── train
│           ├── fgr
│           └── pha
├── seg_vid
│   └── YouTubeVIS
│       └── train
│           ├── JPEGImages
│           └── instances.json
└── seg_img
    ├── coco
    │   ├── train2017
    │   ├── panoptic_train2017
    │   └── annotations
    │       └── panoptic_train2017.json
    └── SuperviselyPersonDataset
        ├── img
        └── seg
```

## Training

*The training code has been simplified for the public release and may contain minor issues. If you encounter any problems, please open an issue.*

```shell
GPU=8
OMP_NUM_THREADS=${GPU} torchrun --master_port 25357 --nproc_per_node=${GPU} matanyone/train.py
```

- Set `GPU` to change the number of GPUs.
- Change `master_port` if you encounter port collision.
- To disable training of `stage_1`, specify `stage_1.enabled=False`. Same for other stages.
- To finetune from a pre-trained model, specify `weights=[path to the weights]`.
- To resume training, specify `checkpoint=[path to the checkpoint]`.
- Models, logs, and visualizations will be saved in `./matting-logs/`.
- The training consists of 3 stages, configured in `./matanyone/config/train_config.yaml`. For details, please refer to our [paper](https://arxiv.org/abs/2501.14677).
    - `use_video`: Train with video matting datasets (`True`) or image matting datasets (`False`). We only train with image matting data at stage 3.
    - `core_supervision`: Add core supervision during training (visualizations in `matting-logs/matanyone_release/stage_x_seg_mat_images`). We start applying core supervision from stage 2.
    