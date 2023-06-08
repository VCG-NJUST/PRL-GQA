# No-reference Point Cloud Geometry Quality Assessment Based on Pairwise Rank Learning

We propose no-reference point-based geometry-only quality assessment framework based on deep pairwise rank learning, named `PRL-GQA`, and a new large-scale pairwise ranking learning dataset called `PRLD`.

# Overview

A pairwise rank learning framework for no-reference point cloud geometry-only quality assessment (`PRL-GQA`) is proposed, in which the geometric feature extraction, weighted quality index calculation, and pairwise rank learning are jointly optimized in an end-to-end manner.

![image](.\fig\overview.png)



A new rank dataset named `PRLD` is constructed, which includes 150 reference point clouds and 15750 pairs of distorted samples.

![image](.\fig\PRLD.png)

Our `PRLD` dataset is released at https://pan.baidu.com/s/1eD_1178tdh_Fg3VuAGiubA?pwd=8jt4.

# Author's Implementations

The experiments in our paper are done with the pytorch implementation.

# Run Program

Please copy the `PRLD` dataset in the program root directory

Run and train the model:

```shell
python train_test.py
```

Verify the performance:

```she	
python perfermance_validate.py
```



## Citation

Please cite this paper if you want to use it in your work,

## License

MIT License

## Acknowledgement

The structure of this codebase is borrowed from [PointNet](https://github.com/charlesq34/pointnet).