# Reinforcement Learning Tasks

## Command
If you want to use PPO with our proposed critic use learn-ppo.py.
If you want to run REINFORCE without critic, use learn-reinforce.py.

## NLRL tasks

Below provides a proper set of parameters for those tasks.
``` shell
# Train: add --nlm-breadth 2 --nlm-depth 3
$ python learn-ppo.py --task nlrl-Stack 
$ python learn-ppo.py --task nlrl-Unstack
$ python learn-ppo.py --task nlrl-On
# Test
$ python learn-ppo.py --task nlrl-TASK --test-only --load $CHECKPOINT
```

## Sorting

Below provides a proper set of parameters for this task.
``` shell
# Train
$ python learn-ppo.py --task sort --curriculum-start 3 --curriculum-graduate 15
# Test
$ python learn-ppo.py --task sort --test-only --load $CHECKPOINT
```

## Shortest path

Below provides a proper set of parameters for this task.
``` shell
# Train
$ python learn-ppo.py --task path --curriculum-start 3 --curriculum-graduate 10
# Test
$ python learn-ppo.py --task path --test-only --load $CHECKPOINT
```

## Blocksworld

``` shell
# Train
$ python learn-ppo.py --task final --curriculum-graduate 12
# Test
$ python learn-ppo.py --task final --test-only --load CHECKPOINT
```
