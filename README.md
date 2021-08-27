# AET-SGD
AET-SGD implementation


In order to run the AET-SGD, you need to execute the following command:

CUDA_VISIBLE_DEVICES=0,1 python3 main_SGD_event.py --num-clients 5  --dataset 0 --eta0 0.01 --num-edge 2 --log-interval 10 --batch-size 1 --event-name EVENT

Here, main_SGD_event.py is the Pytorch impelementation of AET-SGD.

--dataset 0 is the index of MNIST data set.

--num-edge 2 is the number of compute nodes joining the training process.

--log-interval 10 shows the logging event after every 10 SGD iterations.

--batch-size 1 is the size of mini-batch size.

--event-name EVENT is the name of event-triggered SGD.

Note that CUDA_VISIBLE_DEVICES=0,1 indicates which GPU cards you want to run the experiments.
