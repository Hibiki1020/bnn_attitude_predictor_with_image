#!/bin/bash

image_name = "bnn_attitude_predictor_with_image"
tag_name = "nvidia_docker_noetic"
root_path = $(pwd)

# /media/amsl/96fde31e-3b9b-4160-8d8a-a4b913579ca2
# is ssd path in author's environment

xhost +
docker run -it --rm \
	--gpus all \
	--env="DISPLAY" \
	--env="QT_X11_NO_MITSHM=1" \
	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --net=host \
	-v $root_path/../../../bnn_attitude_predictor_with_image:/home/ros_catkin_ws/src/$image_name \
    -v /media/amsl/96fde31e-3b9b-4160-8d8a-a4b913579ca2:/home/ssd_dir \
	$image_name:$tag_name