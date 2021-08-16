wget https://gitlab.com/libeigen/eigen/-/archive/3.4-rc1/eigen-3.4-rc1.zip
unzip eigen-3.4-rc1.zip && rm eigen-3.4-rc1.zip
# shellcheck disable=SC2164
mkdir eigen-3.4-rc1/build && cd eigen-3.4-rc1/build
cmake ..
make install && cd ../../

wget -O opencv.zip https://github.com/opencv/opencv/archive/master.zip
unzip opencv.zip && rm opencv.zip
# shellcheck disable=SC2164
mkdir opencv-master/build && cd opencv-master/build
cmake ..
make install && cd ../../
