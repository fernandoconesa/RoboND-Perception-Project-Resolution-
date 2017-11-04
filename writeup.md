# Exercise 1, 2 and 3 Pipeline Implemented
## Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.

The simulator runs a camera and its information is published in ros. In order to handle this information it is necessary to convert it to PCL

    # TODO: Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)


With the data it is necessary to convert to a voxel grid, which must be configured with the proper size. This simplifies the information, but the correct size of the scene must be found through trial and error.

    # TODO: Voxel Grid Downsampling
    vox = cloud.make_voxel_grid_filter()
    # Choose a voxel (also known as leaf) size

    LEAF_SIZE = 0.005
    # Set the voxel (or leaf) size
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    # Call the filter function to obtain the resultant downsampled point cloud
    cloud_filtered = vox.filter()


In the PCL library it is implemented the RANSAC filter. The table is considered a plane, because of that this is the parameter used as model. 

    # TODO: RANSAC Plane Segmentation
    # Create the segmentation object
    seg = cloud_filtered.make_segmenter()
    # Set the model you wish to fit
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    # Max distance for a point to be considered fitting the model
    # Experiment with different values for max_distance
    # for segmenting the table
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)
    # Call the segment function to obtain set of inlier indices and model coefficients
    inliers, coefficients = seg.segment()

    # TODO: Extract inliers and outliers
    # Extract inliers
    extracted_inliers = cloud_filtered.extract(inliers, negative=False)
    # Extract outliers
    extracted_outliers = cloud_filtered.extract(inliers, negative=True)

This extract the plane (table) from the objects (outliers). 

## Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.

With the separation between objects and table, it is necessary to separate them into clusters.

    # TODO: Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(extracted_outliers)# Apply function to convert XYZRGB to XYZ
    tree = white_cloud.make_kdtree()

    #Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold
    # Your task is to experiment and find values that work for segmenting objects.
    ec.set_ClusterTolerance(0.01)
    ec.set_MinClusterSize(20)
    ec.set_MaxClusterSize(10000)


The Euclidean Clustering algorithm in PCL requires a cloud with only spatial information, so the color information is removed. To separate the different objects into clusters, the parameters (tolerance and cluster size) must be configured. This is another case of trial and error in order to find the correct ones.

    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()    
    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    #Assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                            rgb_to_float(cluster_color[j])])


    #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

This algorithm only supports k-d tree for nearest neighbor search. Cluster_indices contains a list of indices for each cluster and each is represented in different color.

    # TODO: Convert PCL data to ROS messages
    ros_cloud_objects = pcl_to_ros(extracted_outliers)
    ros_cloud_table = pcl_to_ros(extracted_inliers)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # TODO: Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)

the point clouds obtained are converted from PCL to ROS and published.


## Complete Exercise 3 Steps. Features extracted and SVM trained. Object recognition implemented.


    detected_objects_labels = []
    detected_objects = []
    for index, pts_list in enumerate(cluster_indices):
    # Classify the clusters! (loop through each detected cluster one at a time)
        pcl_cluster = extracted_outliers.extract(pts_list)
        # TODO: convert the cluster from pcl to ROS using helper function
        sample_cloud = pcl_to_ros(pcl_cluster)
        # Extract histogram features
        # TODO: complete this step just as is covered in capture_features.py
        chists = compute_color_histograms(sample_cloud, using_hsv=True)
        normals = get_normals(sample_cloud)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))
        #labeled_features.append([feature, model_name])
        # Grab the points for the cluster
        # Compute the associated feature vector
        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)
        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))
        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster_cloud
        detected_objects.append(do)

    print(detected_objects_labels)
    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))


A way to identify the objects is through color histograms and the normals. The features are obtained in the training.launch, within sensor_stick. Running the capture_features.py file, these data is obtained.
With the data of the objects it is necessary to train it through the SVM (Support Vector Machine) learning algorithm. Running the train_svm_py script, the data obtained is trained, with the following results.






With an accuracy over 90% in the three different “worlds” of the simulation.


# Pick and Place Setup



The object recognition part of the code works fine, fulfilling the objectives of the project:


![alt text](https://github.com/fernandoconesa/RoboND-Perception-Project-Resolution-/blob/master/Files/Mundo%20uno%20OK.PNG)




In order to improve the results it could be possible to increase the numbers of samples taken in the capture_features.py script, or trying a different configuration of the cluster extraction.

