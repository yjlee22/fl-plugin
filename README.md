# Embedding Byzantine Fault Tolerance into Federated Learning via Consistency Scoring

This is an official implementation of the following paper:
> Youngjoon Lee, Jinu Gong, and Joonhyuk Kang.
**[Embedding Byzantine Fault Tolerance into Federated Learning via Consistency Scoring](https://arxiv.org/abs/2411.10212)**  
_IEEE GLOBECOM 2025 (Accepted)_.

## Docker Image
`docker pull gaudi-docker/1.17.0/ubuntu22.04/habanalabs/pytorch-installer-2.3.1:latest`

## Requirements
Please install the required packages as below
```pip install tensorboard torch medmnist numpy```

## Dataset
- Blood cell classification dataset ([A dataset of microscopic peripheral blood cell images for development of automatic recognition systems](https://www.sciencedirect.com/science/article/pii/S2352340920303681))

## Federated Learning Techniques
This paper considers the following federated learning techniques
- FedAvg ([McMahan, Brendan, et al. AISTATS 2017](http://proceedings.mlr.press/v54/mcmahan17a?ref=https://githubhelp.com))
- FedProx ([Li, Tian, et al. MLSys 2020](https://proceedings.mlsys.org/paper/2020/hash/38af86134b65d0f10fe33d30dd76442e-Abstract.html))
- FedDyn ([Acar, Durmus Alp Emre, et al. ICLR 2021](https://arxiv.org/abs/2111.04263))
- FedRS ([Li, X. C., & Zhan, D. C. SIGKDD 2021](https://dl.acm.org/doi/10.1145/3447548.3467254))
- FedSAM ([Qu, Zhe, et al. ICML 2022](https://arxiv.org/abs/2206.02618))
- FedSpeed ([Qu, Zhe, et al. ICLR 2023](https://arxiv.org/abs/2302.10429))

## Citation
If this codebase can help you, please cite our paper: 
```bibtex
@article{lee2024embedding,
  title={Embedding Byzantine Fault Tolerance into Federated Learning via Consistency Scoring},
  author={Lee, Youngjoon and Gong, Jinu and Kang, Joonhyuk},
  journal={arXiv preprint arXiv:2411.10212},
  year={2024}
}
```