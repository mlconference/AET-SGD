<h1>AET-SGD implementation using Pytorch</h1>

This repo contains the implementation of ET-SGD (Event-triggered SGD) and AET-SGD (Asynchronous AET-SGD). The main difference here is our AET-SGD deploys the linear increasing sample sequence for p2p network topology while ET-SGD deploys the heuristic event threshold.


<em>In order to run the ET-SGD, you need to execute the following commands:</em>

##### ```CUDA_VISIBLE_DEVICES=0,1 python3 main_SGD_event.py --num-clients 5  --dataset 0 --eta0 0.01 --num-edge 2 --log-interval 10  --eta0 0.01 --batch-size 1 --event-name EVENT```

Here, main_SGD_event.py is the Pytorch implementation of AET-SGD.

```--dataset 0``` is the index of the MNIST data set.

```--num-edge 2``` is the number of compute nodes joining the training process.

```--log-interval 10``` shows the logging event after every 10 SGD iterations.

```--eta0 0.01``` is the initial step size.

```--batch-size 1``` is the size of mini-batch size.

```--event-name EVENT``` is the name of event-triggered SGD. The default event is ```local SGD```.

Note that ```CUDA_VISIBLE_DEVICES=0,1``` indicates which GPU cards you want to run the experiments on.


<em> In order to run the experiment with AET-SGD, you can execute the following commands:</em>

##### ```CUDA_VISIBLE_DEVICES=0,1 python3 main_SGD.py --num-clients 5 --dataset 0 --eta0 0.01 --num-edge 2 --log-interval 10 --batch-size 10 --need-constant-ss 1 --constant-ss 10```

Here, AET-SGD with local SGD runs with a constant local SGD (10 SGD iterations per round) and,

```--need-constant-ss 1``` enables the constant local SGD setting.

```--constant-ss 10``` is the number of local SGD for each commumication round.


<em>Finally, our AET-SGD with linear increasing sample sequences can be executed by </em>

##### ```CUDA_VISIBLE_DEVICES=0,1 python3 main_SGD.py --num-clients 5  --dataset 0 --eta0 0.01 --num-edge 2 --log-interval 10 --batch-size 1```

The default linear increasing sample sequence is ```s_i = 10i```, where i is the current communication round.

