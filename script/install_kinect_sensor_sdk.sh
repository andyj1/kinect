git clone --recursive https://github.com/microsoft/Azure-Kinect-Sensor-SDK.git
cd Azure-Kinect-Sensor-SDK
git checkout release/1.3.x

sudo dpkg --add-architecture amd64
sudo apt update
sudo apt install -y \
    pkg-config \
    ninja-build \
    doxygen \
    clang \
    gcc-multilib \
    g++-multilib \
    python3 \
    nasm

sudo apt install -y \
    libgl1-mesa-dev \
    libsoundio-dev \
    libvulkan-dev \
    libx11-dev \
    libxcursor-dev \
    libxinerama-dev \
    libxrandr-dev \
    libusb-1.0-0-dev \
    libssl-dev \
    libudev-dev \
    mesa-common-dev \
    uuid-dev

ninja-build
sudo apt-get install ninja-build

openssl
sudo apt-get install libssl-dev


x11
sudo apt-get install libx11-dev


randr library
sudo apt-get install xorg-dev libglu1-mesa-dev

mkdir build && cd build
cmake .. -GNinja
ninja

install libk4a1.3 (most
download depth engine file libdepthengine.so.2.0 recent)
link: https://drive.google.com/open?id=1nryM1mghLDAp64F-RMdruirotcDH7U6c
copy to /Azure-Kinect-Sensor-SDK/build/bin

copy rules so one can use without being 'root'
sudo cp ../scripts/99-k4a.rules /etc/udev/rules.d/










