# Exercise 1, 2 and 3 Pipeline Implemented
## Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.

The simulator runs a camera and its information is published in ros. In order to handle this information it is necessary to convert it to PCL

    # TODO: Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)


With the data it is necessary to convert to a voxel grid, which must be configured with the proper size. This simplifies the information, but the correct size of the scene must be found through trial and error.



In the PCL library it is implemented the RANSAC filter. The table is considered a plane, because of that this is the parameter used as model. 



This extract the plane (table) from the objects (outliers). 

## Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.

With the separation between objects and table, it is necessary to separate them into clusters.



The Euclidean Clustering algorithm in PCL requires a cloud with only spatial information, so the color information is removed. To separate the different objects into clusters, the parameters (tolerance and cluster size) must be configured. This is another case of trial and error in order to find the correct ones.




This algorithm only supports k-d tree for nearest neighbor search. Cluster_indices contains a list of indices for each cluster and each is represented in different color.



the point clouds obtained are converted from PCL to ROS and published.


## Complete Exercise 3 Steps. Features extracted and SVM trained. Object recognition implemented.



A way to identify the objects is through color histograms and the normals. The features are obtained in the training.launch, within sensor_stick. Running the capture_features.py file, these data is obtained.
With the data of the objects it is necessary to train it through the SVM (Support Vector Machine) learning algorithm. Running the train_svm_py script, the data obtained is trained, with the following results.






With an accuracy over 90% in the three different “worlds” of the simulation.


# Pick and Place Setup



The object recognition part of the code works fine, fulfilling the objectives of the project:







In order to improve the results it could be possible to increase the numbers of samples taken in the capture_features.py script, or trying a different configuration of the cluster extraction.

