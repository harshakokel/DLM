# Differentiable Logic Machines [PDF](https://arxiv.org/abs/2102.11529)
The integration of reasoning, learning, and decision-making is key to build more general AI systems.
As a step in this direction, we propose a novel neural-logic architecture that can solve both inductive logic programming (ILP) and deep reinforcement learning (RL) problems.
Our architecture defines a restricted but expressive continuous space of first-order logic programs by assigning weights to predicates instead of rules.
Therefore, it is fully differentiable and can be efficiently trained with gradient descent.
Besides, in order to solve more efficiently RL problems, we propose a novel critic architecture that enables actor-critic algorithms.
Compared to state-of-the-art methods on both ILP and RL problems, our proposition achieves excellent performance, while being able to provide a fully interpretable solution and scaling much better, especially during the testing phase.

## Prerequisites
* Python 3
* PyTorch 1.5.1
* [Jacinle](https://github.com/vacancy/Jacinle). 
  We use the version [ed90c3a](https://github.com/vacancy/Jacinle/tree/ed90c3a70a133eb9c6c2f4ea2cc3d907de7ffd57). 
  (to be cloned inside the third_party directory)

## Installation

```
#git clone REPO --recursive
virtualenv3 -p /usr/bin/python3.7 pyDLM
. pyDLM/bin/activate
pip install --upgrade pip
pip install torch==1.5.1 torchvision==0.6.1 six tqdm PyYAML keras tensorflow
```

## Usage

``` shell
PATH_TO_SOURCE=$(pwd)
export PYTHONPATH=$PYTHONPATH:$PATH_TO_SOURCE:$PATH_TO_SOURCE/third_party/Jacinle/

# To train a model:
$ python scripts/rl/learn-ppo.py --task final --model dlm
# To test a model:
$ python scripts/rl/learn-ppo.py --task final --test-only --load models/nlm/blocksworld.pth
```
For more details, please have a look at the dedicated README.md for the [supervised learning case](scripts/supervised/) and [reinforcement learning case](scripts/rl/).

## Cite

If you make use of this code, please cite:

```
@misc{zimmer2021differentiable,
      title={Differentiable Logic Machines}, 
      author={Matthieu Zimmer and Xuening Feng and Claire Glanois and Zhaohui Jiang and Jianyi Zhang and Paul Weng and 
      Li Dong and Hao Jianye and Liu Wulong},
      year={2021},
      eprint={2102.11529},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
``` 

## Acknowledgement

This repository is based on the source code of [Neural Logic Machines](https://github.com/google/neural-logic-machines/) 
by Honghua Dong, Jiayuan Mao, Tian Lin, Chong Wang, Lihong Li, and Denny Zhou.
