#!/bin/bash
# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
#
# Script to build OpenCV from source code and modify the MediaPipe opencv config.
# Note that this script only has been tested on Debian 9 and Ubuntu 16.04.
# usage:
# $ cd <mediapipe root dir>
# $ chmod +x ./setup_opencv.sh
# $ ./setup_opencv.sh
set -e

opencv_build_file="$( cd "$(dirname "$0")" ; pwd -P )"/third_party/opencv_linux.BUILD
echo $opencv_build_file

echo "Installing OpenCV from source"
sudo apt update && sudo apt install build-essential git
sudo apt install cmake ffmpeg libavformat-dev libdc1394-22-dev libgtk2.0-dev \
                 libjpeg-dev libpng-dev libswscale-dev libtbb2 libtbb-dev \
                 libtiff-dev
rm -rf /tmp/build_opencv
mkdir /tmp/build_opencv
cd /tmp/build_opencv
git clone https://github.com/opencv/opencv_contrib.git
git clone https://github.com/opencv/opencv.git
mkdir opencv/release
cd opencv/release
cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_opencv_ts=OFF \
      -DOPENCV_EXTRA_MODULES_PATH=/tmp/build_opencv/opencv_contrib/modules \
      -DBUILD_opencv_aruco=OFF -DBUILD_opencv_bgsegm=OFF -DBUILD_opencv_bioinspired=OFF \
      -DBUILD_opencv_ccalib=OFF -DBUILD_opencv_datasets=OFF -DBUILD_opencv_dnn=OFF \
      -DBUILD_opencv_dnn_objdetect=OFF -DBUILD_opencv_dpm=OFF -DBUILD_opencv_face=OFF \
      -DBUILD_opencv_fuzzy=OFF -DBUILD_opencv_hfs=OFF -DBUILD_opencv_img_hash=OFF \
      -DBUILD_opencv_js=OFF -DBUILD_opencv_line_descriptor=OFF -DBUILD_opencv_phase_unwrapping=OFF \
      -DBUILD_opencv_plot=OFF -DBUILD_opencv_quality=OFF -DBUILD_opencv_reg=OFF \
      -DBUILD_opencv_rgbd=OFF -DBUILD_opencv_saliency=OFF -DBUILD_opencv_shape=OFF \
      -DBUILD_opencv_structured_light=OFF -DBUILD_opencv_surface_matching=OFF \
      -DBUILD_opencv_world=OFF -DBUILD_opencv_xobjdetect=OFF -DBUILD_opencv_xphoto=OFF
make -j 16
sudo make install
rm -rf /tmp/build_opencv
echo "OpenCV has been built. You can find the header files and libraries in /usr/local/include/opencv4/opencv2 and /usr/local/lib"

# Modify the build file.
echo "Modifying MediaPipe opencv config"
sed -i "s/lib\/x86_64-linux-gnu/local\/lib/g" $opencv_build_file
sed -i "20i \            \"local/lib/libopencv_optflow.so*\"," $opencv_build_file
sed -i "/includes =/d" $opencv_build_file
sed -i "/hdrs =/d" $opencv_build_file
line_to_insert=$(grep -n 'linkstatic =' $opencv_build_file | awk -F  ":" '{print $1}')'i'
sed -i "$line_to_insert \   includes = [\"local\/include\/opencv4\/\"]," $opencv_build_file
sed -i "$line_to_insert \   hdrs = glob([\"local\/include\/opencv4\/\*\*\/\*\.h\*\"])," $opencv_build_file

echo "Done"
