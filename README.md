# WonderHuman: Hallucinating Unseen Parts in Dynamic 3D Human Reconstruction

[Project Page](https://wyiguanw.github.io/WonderHuman/) | [Paper]() | [Video](https://youtu.be/bdwUL_RKajA)

This is an official implementation. The codebase is implemented using [PyTorch](https://pytorch.org/) and tested on [Ubuntu](https://ubuntu.com/) 20.04.6 LTS.
## Todo

- [ ] Release the reorganized code.
- [ ] Provide the scripts for data preprocessing.
- [ ] Provide the code for free-view rendering and annimation.
## Prerequisite

### `Configure environment`

Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended) or [Anaconda](https://www.anaconda.com/).

Create and activate a virtual environment.

    conda env create --file environment.yml
	conda activate WonderHuman

Install Pytorch. This codebase is tested on CUDA 11.8, Pytorch 2.0.1:

    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

Then, compile ```diff-gaussian-rasterization``` and ```simple-knn``` as in [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) repository.
   
    pip install submodules/diff-gaussian-rasterization
    pip install submodules/simple-knn
    
Install the required packages.

    pip install -r requirements.txt

### `Download SMPL model`

- SMPL/SMPL-X model: register and download [SMPL](https://smpl.is.tue.mpg.de/) and [SMPL-X](https://smpl-x.is.tue.mpg.de/), and put these files in ```assets/smpl_files```. The folder should have the following structure:
```
smpl_files
 └── smpl
   ├── SMPL_FEMALE.pkl
   ├── SMPL_MALE.pkl
   └── SMPL_NEUTRAL.pkl
 └── smplx
   ├── SMPLX_FEMALE.npz
   ├── SMPLX_MALE.npz
   └── SMPLX_NEUTRAL.npz
```
To be continue ...

## Run on Your Own Video

### Preprocessing

- masks and poses: use the bash script `scripts/custom/process-sequence.sh` in [InstantAvatar](https://github.com/tijiang13/InstantAvatar). The data folder should have the followings:
```
smpl_files
 ├── images
 ├── masks
 ├── cameras.npz
 └── poses_optimized.npz
```
- data format: we provide a script to convert the pose format of romp to ours (remember to change the `path` in L50 and L51):
```
cd scripts & python sample_romp2gsavatar.py
```
- position map of the canonical pose: (remember to change the corresponding `path`)
```
python gen_pose_map_cano_smpl.py
```
### Training for Stage 1

```
cd .. &  python train.py -s $path_to_data/$subject -m output/{$subject}_stage1 --train_stage 1 --pose_op_start_iter 10
```

### Training for Stage 2

- export predicted smpl:
```
cd scripts & python export_stage_1_smpl.py
```
- visualize the optimized smpl (optional):
```
python render_pred_smpl.py
```
- generate the predicted position map:
```
python gen_pose_map_our_smpl.py
```
- start to train
```
cd .. &  python train.py -s $path_to_data/$subject -m output/{$subject}_stage2 --train_stage 2 --stage1_out_path $path_to_stage1_net_save_path
```


## Acknowledgement

The implementation took reference from [GaussianAvatar](https://github.com/aipixel/GaussianAvatar), [threestudio](https://github.com/threestudio-project/threestudio). We thank the authors for their generosity to release code.

## Citation

If you find our work useful, please consider citing:

```BibTeX

```
