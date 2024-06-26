# Scheduling autonomous robots for an intralogistic application: A comparison of lookahead-based ADP strategies

## Brief description
#### The problem and the objective 
Imagine a robot employed in a warehouse, where it must pick objects and transport them to designated destinations, i.e., trays.
The robot can indeed perform three action types: move in the warehouse, pick and throw an object. Moving and throwing risks are associated to the corresponding actions.
The objective is to find a time-efficient and risk-aware optimal scheduling of tasks, when multiple orders with different priorities arrive into the system following a Poisson process.

#### Setting
A completely connected graph is used to model the setting; its nodes are either picking or throwing locations and the robot can only move along the graph's edges, which contain information on moving risk and travelling time. The graph is the result of a previously solved routing optimization problem. 

#### Resolution method
The Dynamic Programming (DP) paradigm and some approximate versions of its are chosen as resolution methods. In particular, the work delves into the comparison between two Lookahead Policies: Myopic Rollout (MR) and Monte-Carlo Tree Search (MCTS) are ideed implemented.

## Set up
#### Data
Data are found in 
```bash 
/input_data
```
folder. There are mainly pkl files containing info on the problem setting: graph, nodes, arches; and info on objects and trays.

#### Files

Data are uploaded in all files where needed. Main source files are ExactDP.py, MR.py and MCTS.py.

## Aknowledgement
This work has been partially funded by the European Union’s Horizon 2020 research and innovation programme, under grant agreement No 101017274.
See https://darko-project.eu/about/ for more information about the DARKO project.
