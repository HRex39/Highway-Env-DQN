# Highway-Env-DQN
Highway-Env Agent using DQN  

## PreRequisite
1. [Highway-Env](https://github.com/eleurent/highway-env) or ```pip install highway-env```
2. [Pytorch](https://pytorch.org/)
3. Tensorboard(pip install tensorboard)

## Usage
1. train
```
python3 train_cpu.py
tensorboard --logdir=./log --port <your port>
```

2. play
```
<change loading path in play.py>
python3 play.py
```
