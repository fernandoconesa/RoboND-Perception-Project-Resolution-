3D Perception

Before starting any work on this project, please complete all steps for Exercise 1, 2 and 3. At the end of Exercise-3 you have a pipeline that can identify points that belong to a specific object.

In this project, you must assimilate your work from previous exercises to successfully complete a tabletop pick and place operation using PR2.

The PR2 has been outfitted with an RGB-D sensor much like the one you used in previous exercises. This sensor however is a bit noisy, much like real sensors.

Given the cluttered tabletop scenario, you must implement a perception pipeline using your work from Exercises 1,2 and 3 to identify target objects from a so-called “Pick-List” in that particular order, pick up those objects and place them in corresponding dropboxes.
Project Setup

For this setup, catkin_ws is the name of active ROS Workspace, if your workspace name is different, change the commands accordingly If you do not have an active ROS workspace, you can create one by:

$ mkdir -p ~/catkin_ws/src
$ cd ~/catkin_ws/
$ catkin_make

Now that you have a workspace, clone or download this repo into the src directory of your workspace:

$ cd ~/catkin_ws/src
$ git clone https://github.com/udacity/RoboND-Perception-Project.git

Note: If you have the Kinematics Pick and Place project in the same ROS Workspace as this project, please remove the 'gazebo_grasp_plugin' directory from the RoboND-Perception-Project/ directory otherwise ignore this note.

Now install missing dependencies using rosdep install:

$ cd ~/catkin_ws
$ rosdep install --from-paths src --ignore-src --rosdistro=kinetic -y

Build the project:

$ cd ~/catkin_ws
$ catkin_make

Add following to your .bashrc file

export GAZEBO_MODEL_PATH=~/catkin_ws/src/RoboND-Perception-Project/pr2_robot/models:$GAZEBO_MODEL_PATH

If you haven’t already, following line can be added to your .bashrc to auto-source all new terminals

source ~/catkin_ws/devel/setup.bash

To run the demo:

$ cd ~/catkin_ws/src/RoboND-Perception-Project/pr2_robot/scripts
$ chmod u+x pr2_safe_spawner.sh
$ ./pr2_safe_spawner.sh

demo-1

Once Gazebo is up and running, make sure you see following in the gazebo world:

    Robot

    Table arrangement

    Three target objects on the table

    Dropboxes on either sides of the robot

If any of these items are missing, please report as an issue on the waffle board.

In your RViz window, you should see the robot and a partial collision map displayed:

demo-2

Proceed through the demo by pressing the ‘Next’ button on the RViz window when a prompt appears in your active terminal

The demo ends when the robot has successfully picked and placed all objects into respective dropboxes (though sometimes the robot gets excited and throws objects across the room!)

Close all active terminal windows using ctrl+c before restarting the demo.

You can launch the project scenario like this:

$ roslaunch pr2_robot pick_place_project.launch

Required Steps for a Passing Submission:

    Extract features and train an SVM model on new objects (see pick_list_*.yaml in /pr2_robot/config/ for the list of models you'll be trying to identify).
    Write a ROS node and subscribe to /pr2/world/points topic. This topic contains noisy point cloud data that you must work with.
    Use filtering and RANSAC plane fitting to isolate the objects of interest from the rest of the scene.
    Apply Euclidean clustering to create separate clusters for individual items.
    Perform object recognition on these objects and assign them labels (markers in RViz).
    Calculate the centroid (average in x, y and z) of the set of points belonging to that each object.
    Create ROS messages containing the details of each object (name, pick_pose, etc.) and write these messages out to .yaml files, one for each of the 3 scenarios (test1-3.world in /pr2_robot/worlds/). See the example output.yaml for details on what the output should look like.
    Submit a link to your GitHub repo for the project or the Python code for your perception pipeline and your output .yaml files (3 .yaml files, one for each test world). You must have correctly identified 100% of objects from pick_list_1.yaml for test1.world, 80% of items from pick_list_2.yaml for test2.world and 75% of items from pick_list_3.yaml in test3.world.
    Congratulations! Your Done!

Extra Challenges: Complete the Pick & Place

    To create a collision map, publish a point cloud to the /pr2/3d_map/points topic and make sure you change the point_cloud_topic to /pr2/3d_map/points in sensors.yaml in the /pr2_robot/config/ directory. This topic is read by Moveit!, which uses this point cloud input to generate a collision map, allowing the robot to plan its trajectory. Keep in mind that later when you go to pick up an object, you must first remove it from this point cloud so it is removed from the collision map!
    Rotate the robot to generate collision map of table sides. This can be accomplished by publishing joint angle value(in radians) to /pr2/world_joint_controller/command
    Rotate the robot back to its original state.
    Create a ROS Client for the “pick_place_routine” rosservice. In the required steps above, you already created the messages you need to use this service. Checkout the PickPlace.srv file to find out what arguments you must pass to this service.
    If everything was done correctly, when you pass the appropriate messages to the pick_place_routine service, the selected arm will perform pick and place operation and display trajectory in the RViz window
    Place all the objects from your pick list in their respective dropoff box and you have completed the challenge!
    Looking for a bigger challenge? Load up the challenge.world scenario and see if you can get your perception pipeline working there!

For all the step-by-step details on how to complete this project see the RoboND 3D Perception Project Lesson Note: The robot is a bit moody at times and might leave objects on the table or fling them across the room :D As long as your pipeline performs succesful recognition, your project will be considered successful even if the robot feels otherwise!
