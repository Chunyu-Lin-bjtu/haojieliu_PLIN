# PLIN: A Network for Pseudo-LiDAR Point Cloud Interpolation

## Notes
Our network is trained with the KITTI dataset alone.

## Requirements
This code was tested with Python 3 and PyTorch 1.0 on Ubuntu 16.04.
- Install [PyTorch](https://pytorch.org/get-started/locally/) on a machine with CUDA GPU.
- The code for self-supervised training requires [OpenCV](http://pytorch.org/) along with the contrib modules. For instance,

- Download the [KITTI Depth](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) Dataset and the corresponding RGB images. 
- The code, data and result directory structure is shown as follows
```
.
├── self-supervised-depth-completion
├── data
|   ├── kitti_depth
|   |   ├── train
|   |   ├── val_selection_cropped
|   └── kitti_rgb
|   |   ├── train
|   |   ├── val_selection_cropped
├── results
```

## Training and testing
A complete list of training options is available with 
```bash
python main.py -h
```
For instance,
```bash
python main.py --train-mode dense -b 1 # train with the KITTI semi-dense annotations and batch size 1
python main.py --resume [checkpoint-path] # resume previous training
python main.py --evaluate [checkpoint-path] # test the trained model
```
python main.py --evaluate ./model_best.pth.tar

## references
```
[1] @article{liu2020plin,
    title={Plin: A network for pseudo-lidar point cloud interpolation},
    author={Liu, Haojie and Liao, Kang and Lin, Chunyu and Zhao, Yao and Liu, Meiqin},
    journal={Sensors},
    volume={20},
    number={6},
    pages={1573},
    year={2020},
    publisher={Multidisciplinary Digital Publishing Institute}
    }
```
