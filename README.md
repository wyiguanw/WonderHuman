# WonderHuman: Hallucinating Unseen Parts in Dynamic 3D Human Reconstruction (Accepted by TVCG)

[Project Page](https://wyiguanw.github.io/WonderHuman/) | [Paper]() | [Video](https://youtu.be/bdwUL_RKajA)

This is an official implementation. The codebase is implemented using [PyTorch](https://pytorch.org/) and tested on [Ubuntu](https://ubuntu.com/) 20.04.6 LTS.
## Todo

- [ ] Release the reorganized code.
- [ ] Provide the full scripts for data preprocessing.
- [ ] Provide the code for free-view rendering and animation.
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

### Download required models

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
`TO DO: provide a download link for the required data.`



## Acknowledgement

The implementation took reference from [GaussianAvatar](https://github.com/aipixel/GaussianAvatar), [threestudio](https://github.com/threestudio-project/threestudio). We thank the authors for their generosity to release code.

## Citation

If you find our work useful, please consider citing:

```BibTeX
@misc{wang2025wonderhuman,
      title={WonderHuman: Hallucinating Unseen Parts in Dynamic 3D Human Reconstruction}, 
      author={Zilong Wang and Zhiyang Dou and Yuan Liu and Cheng Lin and Xiao Dong and Yunhui Guo and Chenxu Zhang and Xin Li and Wenping Wang and Xiaohu Guo},
      year={2025},
      eprint={2502.01045},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.01045}, 
      }
```
