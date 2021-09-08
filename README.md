# mnist-distributed-problem
This is a super small repository demonstrating a Problem with DistributedDataParallel (DDP) from pytorch. 
A three layer feed forward neural network is trained with MNIST with and without data parallel with the same hyper parameters.  
If you configure DistributedDataParalell to use only one node, the model is quite worse in accuracy. 
If you have any suggestions how to make them equal beside tuning the learning rate please commend or send PR!

I tested it on several plattforms (macOS and ubuntu) with different GPUs or CPUs. Concretely a MacBook Pro with CPU, three ubuntu machines (one with a 1080-ti, one with a 2080-ti and one cluster with 8x P100s), it always gives me this problem that accuracy is worse with the same basis code and hyperparameters when training on a single node with or without DDP. If limited to a single node the expectation would be that they give same results.

Usage:
You need a recent pytorch environment.

Then you can just call the script without DDP:
`python mnist-plain.py`

On the other hand you could call it with DDP the following way to use only one GPU/CPU:
`python mnist-distributed.py -n 1 -g 1 -e 5`

The expectation would be that this gives the same results, but it does not
