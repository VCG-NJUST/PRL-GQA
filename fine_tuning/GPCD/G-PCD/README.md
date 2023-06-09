

Evangelos Alexiou
evangelos.alexiou@epfl.ch


# G-PCD: Geometry Point Cloud Dataset


A series of studies is conducted [1-5] to investigate the performance of state-of-the-art objective quality metrics and propose new subjective and objective evaluation methodologies. For this purpose, a representative set of geometry-only point clouds is assembled and degraded using two different types of distortions.

In this [webpage](https://www.epfl.ch/labs/mmspg/downloads/geometry-point-cloud-dataset/), we make publicly available a dataset consisting of reference point cloud models, degraded stimuli, and subjective quality scores that were obtained in two experimental setups [1,2].


## Contents

Bunny and dragon are selected from the [Stanford 3D Scanning Repository](http://graphics.stanford.edu/data/3Dscanrep/) to represent contents with regular geometry and reduced amount of noise. Cube and sphere are artificially generated using mathematical formulas and represent synthetic contents with highly regular geometry. Finally, vase, is a 3D content manually captured using the [Intel RealSense R200](https://ark.intel.com/content/www/us/en/ark/products/92256/intel-realsense-camera-r200.html) camera, and constitutes a representative point cloud with irregular structure that can be acquired by low-cost depth sensors. To form a representative dataset, the contents are selected considering the following properties:

1. Simplicity, as it would have been difficult for subjects to clearly perceive a complex scene in the absence of texture. Although simple, the complexity of the contents covers a reasonable range.
2. Diversity of geometric structure, as different artifacts may be observed by applying different types of degradations. Thus, the point clouds are generated by different means.
3. Similarity of points density, as the visual quality of point clouds is directly affected by the number of points used to represent an object.


## Types of degradation

The contents are scaled to fit in a bounding box of size 1. Two different types of degradations are applied: (a) Octree-pruning, and (b) Gaussian noise.

a. Octree-pruning is obtained by setting a desirable Level of Details (LoD) value in the Octree that encloses the content; thus, structural loss due to point removal and displacement is observed. The LoD is set appropriately for each content to achieve target percentages of remaining points (ρ), allowing an acceptable deviation of ±2% (ρ = {90%, 70%, 50%, 30%}). 
b. Gaussian noise is used to simulate position errors. The coordinates of every point of the content are modified across X, Y and Z axis, following a target standard deviation (σ = {0.0005, 0.002, 0.008, 0.016}) to account for different levels of noise.


## Data format

The following naming convention is adopted in the provided dataset:

- contentName = {bunny, cube, dragon, sphere, vase},
- degradationType = {D01, D02}, with D01 indicating Octree-pruning and D02 indicating Gaussian noise,
- degradationLevel = {L01, L02, L03, L04}, with increasing numbers indicating higher levels of degradations. For instance, L01 and L04 for D01 correspond to 90% and 30% of remaining points, respectively. L01 and L04 for D02 correspond to 0.005 and 0.016 standard deviation, respectively.


## Download

In the provided [URL link](https://www.epfl.ch/labs/mmspg/wp-content/uploads/2020/06/G-PCD.zip) you can find the dataset which consists of the point cloud stimuli that are generated and used, along with subjective scores that are collected in two experimental setups. The material is placed under folders that follow corresponding naming convention.


### Stimuli

In this folder, you can find the stimuli that are used in the studies [1-5]. Both the reference and the distorted models are given in PLY format using ASCII representation. The reference stimuli are noted as: contentName.ply, while the distorted stimuli are noted as: contentName_degradationType_degradationLevel.ply.


### Subjective scores

In this folder, you can find two sub-folders, namely, "desktop setup" and "hmd ar setup" that contain .csv files with the scores obtained from studies [1] and [2], respectively. In the provided files, the reference stimuli are indicated as: contentName_degradationType_hidden.ply, while the distorted stimuli are noted as: contentName_degradationType_degradationLevel.ply.


#### Desktop setup

These scores are collected from subjective evaluation experiments in a desktop setup, conducted in the framework of [1]. There are two different subjective evaluation methodologies that are adopted, namely, ACR and DSIS. The two sets can be distinguished by the corresponding suffix in the provided .csv files. For further details regarding the setup of the experiment and the results, please refer to [1]. Note, that the same set of scores are also used in [3-5].


#### HMD AR setup

These scores are collected from subjective evaluation experiments in an Augmented Reality (AR) scenario using a head-mounted display (HMD), conducted in the framework of [2]. In this experiment, the DSIS subjective evaluation methodology is adopted. For further details regarding the setup of the experiment and the results, please refer to [2]. It should be noted that the same set of scores is also used in [3].


## Conditions of use

- If you wish to use the provided 'stimuli', or the subjective scores from the 'desktop setup' in your research, we kindly ask you to cite [1].
- If you wish to use the subjective scores from the 'HMD AR setup' in your research, we kindly ask you to cite [2].


## References

[1] E. Alexiou and T. Ebrahimi, "On the performance of metrics to predict quality in point cloud representations," Proc. SPIE 10396, Applications of Digital Image Processing XL, 103961H (19 September 2017). doi: 10.1117/12.2275142

[2] E. Alexiou, E. Upenik and T. Ebrahimi, "Towards subjective quality assessment of point cloud imaging in augmented reality," 2017 IEEE 19th International Workshop on Multimedia Signal Processing (MMSP), Luton, 2017, pp. 1-6. doi: 10.1109/MMSP.2017.8122237

[3] E. Alexiou and T. Ebrahimi, "Impact of Visualisation Strategy for Subjective Quality Assessment of Point Clouds," 2018 IEEE International Conference on Multimedia & Expo Workshops (ICMEW), San Diego, CA, 2018, pp. 1-6. doi: 10.1109/ICMEW.2018.8551498

[4] E. Alexiou and T. Ebrahimi, "Benchmarking of Objective Quality Metrics for Colorless Point Clouds," 2018 Picture Coding Symposium (PCS), San Francisco, CA, 2018, pp. 51-55. doi: 10.1109/PCS.2018.8456252

[5] E. Alexiou and T. Ebrahimi, "Point Cloud Quality Assessment Metric Based on Angular Similarity," 2018 IEEE International Conference on Multimedia and Expo (ICME), San Diego, CA, 2018, pp. 1-6. doi: 10.1109/ICME.2018.8486512
