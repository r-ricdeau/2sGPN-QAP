## Two-stage GPN

### Reference
Iida, Satoko, and Ryota Yasudo. "Solving the QAP by Two-Stage Graph Pointer Networks and Reinforcement Learning." arXiv preprint arXiv:2404.00539 (2024).

### Usage
#### Training
```
python3 train.py <density of trained data (matrices)>
```
For dense QAP, <density of trained data (matrices)> should be 1.0, while it should be smaller for sparse QAP.

#### Test
For example, the following solves had12 instance:
```
python3 test.py ./model/model1.pt ./model/model2.pt ./qaplib/had12.dat
```

### Instances
This repository contains instances derived from QAPlib.
QAPlib：https://coral.ise.lehigh.edu/data-sets/qaplib/ 


