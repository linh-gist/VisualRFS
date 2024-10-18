# DOCKER COMMANDS
# docker build --no-cache -t linhma/visualrfs .
# docker push linhma/visualrfs
# Removing images: docker rmi $(docker images -a -q)
# Stop all the containers: docker stop $(docker ps -a -q)
# Remove all the containers: docker rm $(docker ps -a -q)

# HOW TO USE?
# 1. Running Docker:  docker run -d -p "36901:6901" --name quick --hostname quick linhma/visualrfs
# 2. Openning Browser: http://localhost:36901/vnc.html?password=headless => choose 'noVNC Full Client' => password 'headless'
# 3. Refer: https://accetto.github.io/user-guide-g3/quick-start/

# Use the base image
FROM accetto/ubuntu-vnc-xfce-chromium-g3:20.04

# Switch to root user if necessary
USER root

# Set the working directory
WORKDIR /app

# Download and install Miniconda
COPY ./Miniconda3-latest-Linux-x86_64.sh ./
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda

# Update PATH
ENV PATH="/opt/miniconda/bin:${PATH}"

# Install build essentials including g++, make
RUN apt-get update && apt-get install -y \
    build-essential g++ make cmake \
    python3-distutils \
    libgl1-mesa-glx \
    libglib2.0-0

# Initialize Conda
RUN conda init bash
RUN conda install -y python=3.8

# Copy your code into the container
COPY ./VisualRFS ./VisualRFS
COPY ./eigen-3.4.0 ./eigen-3.4.0
COPY ./MOT16 ./MOT16

# Install required Python packages directly in the base Conda environment
RUN pip install "setuptools<65" && \
    pip install numpy==1.21.6 && \
    pip install h5py==3.1.0 && \
    pip install motmetrics==1.4.0 && \
    pip install pandas==1.3.5 && \
    pip install matplotlib==3.4.1 && \
    pip install packaging==24.0 && \
    pip install protobuf==4.24.4 && \
    pip install opencv-python==3.4.18.65 && \
    pip install lap==0.4.0 && \
    pip install openpyxl==3.0.10

# Build c++ packages project
RUN bash -c "cd VisualRFS/src/cpputils/fuzzylite/fuzzylite/ && ./build.sh && cd ../../ && python setup.py build develop"

# Activate the environment by default (optional)
RUN echo "source activate base" >> ~/.bashrc

# Specify the command to run your application (if needed)
CMD ["bash"]