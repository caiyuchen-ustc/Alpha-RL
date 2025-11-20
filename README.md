# On Predictability of Reinforcement Learning Dynamics for Large Language Models

This repository corresponds to the paper:

> **On Predictability of Reinforcement Learning Dynamics for Large Language Models**  
> *https://arxiv.org/pdf/2510.00553*

---


## Installation

```bash
# clone codebase
git clone https://github.com/caiyuchen-ustc/Alpha-RL.git && cd Alpha-RL

# prepare environment
conda create -y -n AlphaRL python=3.11
conda activate AlphaRL

# install dependencies
pip install -r requirements.txt
```

#
## Download_Model

You can access the checkpoint at the following link: [Hugging Face - caiyuchen](https://huggingface.co/caiyuchen)



```bash
# run
cd eval
sh download_hf.sh
```

## Singular Value Decomposition

```bash
sh svd.sh # Obtain the SVD decomposition of each matrix in a model
```
## Obtain a Rank-k Model

```bash
sh upd_rank.sh 
```

## Model Evaluation
```bash
sh reasoning_eval.sh
```

## t-SNE Visualization of Training Trajectories
```bash
cd analysis #eval/analysis
sh extract_rank1_u.sh #Extract U[:,1]
sh visualize_rank1_u_tsne.sh
```

## PLS (Partial Least Squares) Trajectory Fitting
```bash
sh AlphaPLS.sh
```

## AlphaRL Predict
```bash
sh AlphaPredVector.sh
sh AlphaRLBuildPredictModel.sh
```


This repository provides an evaluation framework inspired by **LIMO**, which can be found [here](https://github.com/GAIR-NLP/LIMO).

If you find this project interesting, feel free to ⭐ star the repository or open an issue for discussion!

If you use this code in your research, please cite:
```bibtex
@article{cai2025predictability,
  title={On Predictability of Reinforcement Learning Dynamics for Large Language Models},
  author={Cai, Yuchen and Cao, Ding and Xu, Xin and Yao, Zijun and Huang, Yuqing and Tan, Zhenyu and Zhang, Benyi and Liu, Guiquan and Fang, Junfeng},
  journal={arXiv preprint arXiv:2510.00553},
  year={2025}
}