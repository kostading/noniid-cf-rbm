# noniid-cf-rbm
Source code used to perform the experiments from ICML paper - "A non-IID Framework for Collaborative Filtering with Restricted Boltzmann Machines".

Read the paper at:

http://proceedings.mlr.press/v28/georgiev13.html

The code is implemented on top of the jaRBM_v1.0 open source project, which needs to be downloaded separately.

Some guidelines:

1. You need to extract the MovieLens 1M dataset into /CineScope/ml-data_1/

Note that there's a file u.info, which specifies the number of users and items.

2. Then you can run CfModelTestExecutor class' main method to run the tests. It will use testCfModelWithCVBig(..) for tests on the 1M dataset, but you can also use testCfModelWithCV(..) to test on the smaller 100k dataset.

The code is not very clean as we didn't have much time to structure and refactor it, but you should be able to understand it and play with the different parameters, e.g. epochs, momentum, weight decay, etc.
