# Graph Tasks

A set of graph-related reasoning tasks ans sorting task.

## Graph tasks

### Family tree tasks
``` shell
# Train: add --train-number 20 --test-number-begin 20 --test-number-step 20 --test-number-end 100
$ python learn-graph-tasks.py --task has-father --epochs 50
$ python learn-graph-tasks.py --task has-sister --epochs 50
$ python learn-graph-tasks.py --task grandparents --epochs 100
$ python learn-graph-tasks.py --task uncle
$ python learn-graph-tasks.py --task maternal-great-uncle --nlm-depth 8
# Test
$ python learn-graph-tasks.py --task TASK --test-only --load $CHECKPOINT
# Test with extracted formulas for speeding up:
$ python learn-graph-tasks.py --task TASK --test-only --load $CHECKPOINT --extract-path
```
### General graph tasks
``` shell
# Train
# AdjacentToRed
$ python learn-graph-tasks.py --task adjacent
# 4-Connectivity
$ python learn-graph-tasks.py --task connectivity
# 6-Connectivity
$ python learn-graph-tasks.py --task connectivity --connectivity-dist-limit 6 --nlm-depth 8 --gen-graph-pmin 0.1 --gen-graph-method dnc
# 1-Outdegree
$ python learn-graph-tasks.py --task outdegree --outdegree-n 1
# 2-Outdegree
$ python learn-graph-tasks.py --task outdegree --nlm-depth 6 --nlm-breadth 4

# Test
$ python learn-graph-tasks.py --task TASK --test-only --load $CHECKPOINT --nlm-depth DEPTH --nlm-breadth BREADTH
# Test with extracted formulas for speeding up:
$ python learn-graph-tasks.py --task TASK --test-only --load $CHECKPOINT --extract-path
```

