# Use a base image with CUDA drivers
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu20.04

ENV CUDA_PATH=/usr/local/cuda
ENV CUDA_INCLUDE_PATH=$CUDA_PATH/include
ENV CUDA_LIB_PATH=$CUDA_PATH/lib64

ENV DEBIAN_FRONTEND=noninteractive

# Install Ubuntu packages
RUN apt-get update -y 
RUN apt-get install software-properties-common -y 
RUN add-apt-repository -y multiverse 
RUN apt-get update -y && apt-get upgrade -y 
RUN apt-get install -y apt-utils nano man build-essential wget sudo
# RUN rm -rf /var/lib/apt/lists/*


# Create the /scratch directory
RUN mkdir /scratch
# Copy the entire project folder to the container. Not needed
COPY . /scratch 
#Remove onpolicy folder since we use a local copy in CSF
RUN rm -rf onpolicy 

# Set the working directory
WORKDIR /scratch



# Install curl and other dependencies
RUN apt-get update -y && apt-get install -y curl libssl-dev openssl libopenblas-dev \
    libhdf5-dev hdf5-helpers hdf5-tools libhdf5-serial-dev libprotobuf-dev protobuf-compiler git

# Install dependencies from requirements.txt

RUN apt-get update && apt-get install -y python3 python3-pip unzip
RUN rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip

# Unzip SMAC and export to environment
# This does need to be moved to the /scratch directory

RUN chmod +x install_sc.sh && bash install_sc.sh
 

RUN pip3 install -r requirements.txt
# RUN pip3 install -U ray

RUN mkdir /scratch/on-policy
WORKDIR /scratch/on-policy
ENV PYTHONPATH "/scratch/on-policy:${PYTHONPATH}"
