# ubunutu is the base image
FROM ubuntu:20.04


MAINTAINER Artemiy Shirokov <tema77078@gmail.com>


# this is for timezone config
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update
#-y is for accepting yes when the system asked us for installing the package
RUN apt-get install -y build-essential cmake git openssh-server gdb pkg-config valgrind systemd-coredump

# 1) POCO
RUN echo "************************ POCO ************************"
RUN apt-get install -y openssl libssl-dev
RUN apt-get -y update && apt-get -y install git g++ make cmake libssl-dev
RUN git clone -b master https://github.com/pocoproject/poco.git
WORKDIR poco
RUN mkdir cmake-build
WORKDIR cmake-build
RUN cmake ..
RUN cmake --build . --config Release
RUN cmake --build . --target install
RUN mkdir install
RUN cmake -DCMAKE_INSTALL_PREFIX=$(pwd)/install .. && cmake --build . --target install
WORKDIR "/"

# 2)  OpenCV
RUN echo "************************ OpenCV ************************"
RUN apt update -y && apt install -y cmake g++ wget unzip
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
RUN unzip opencv.zip
RUN mkdir -p build
WORKDIR build
RUN cmake  ../opencv-4.x
RUN make -j4
RUN make install
WORKDIR "/"

# 3)  ONNXRuntime
RUN echo "************************ ONNXRuntime ************************"
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.13.1/onnxruntime-linux-x64-1.13.1.tgz
RUN tar -zxvf onnxruntime-linux-x64-1.13.1.tgz
RUN rm onnxruntime-linux-x64-1.13.1.tgz
WORKDIR "/"
