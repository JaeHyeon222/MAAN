# MAAN

## The Environments

The evaluation environments adopted by us is recorded in the `requirements.txt`. After you built your own basic Python (Python = 3.9 in our setting) setup via either *virtual environment* or *anaconda*, please try to keep similar to it via:

- Step1: install Pytorch first:
`pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117`

- Step2: install other libs via:
```pip install -r requirements.txt```

or take it as a reference based on your original environments.

## The Validation datasets
download all the necessary validate dataset ([DIV2K_LSDIR_valid_LR](https://drive.google.com/file/d/1YUDrjUSMhhdx1s-O0I1qPa_HjW-S34Yj/view?usp=sharing) and [DIV2K_LSDIR_valid_HR](https://drive.google.com/file/d/1z1UtfewPatuPVTeAAzeTjhEGk4dg2i8v/view?usp=sharing))

## How to test the baseline model?

1. `git clone https://github.com/JaeHyeon222/MAAN.git`
2. Select the model you would like to test from [`run.sh`](./run.sh)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python test_demo.py --data_dir [path to your data dir] --save_dir [path to your save dir] --model_id 29
    ```
    - Be sure the change the directories `--data_dir` and `--save_dir`.
3. More detailed example-command can be found in `run.sh` for your convenience.

As a reference, we provide the results of MAAN below:
- Average PSNR on DIV2K_LSDIR_valid: 27.01 dB
- Average PSNR on DIV2K_LSDIR_test: 27.13 dB
- Number of parameters: 0.168 M
- Runtime: - ms (Average runtime of - ms on DIV2K_LSDIR_valid data and - ms on DIV2K_LSDIR_test data)
- FLOPs on an LR image of size 256×256: 10.62 G

## Our result test datsets
[DIV2K_LSDIR_test_SR](https://drive.google.com/file/d/1FmggUV1-kepcvv5SDcb1cKOJa2hU3NPu/view?usp=drive_link)
