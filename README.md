# ProtoGNN

This repository is the implementation of [ProtoGNN: Prototype-assisted Message Passing Framework for Non-Homophilous Graphs](). 

## Requirements

1. Pytorch-1.11.0
2. Pytorch-geometric-2.0.4
3. Sacred 0.8.2
4. Cuda toolkit-11.3

Other higher versions will probably work as well. Make relevant changes during installation for other versions

## Training and Evaluation

Example:

```
python main.py with data.dataset=cornell5 optim.ortho_weight=0.1 optim.comp_weight=0.01 proto.num=10
```


This code is based on [pytorch-geometric](https://github.com/rusty1s/pytorch_geometric)
