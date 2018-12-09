FROM nvidia/cuda:10.0-devel-ubuntu16.04

ENV USERNAME simtrack
ENV WORKSPACE /home/$USERNAME/my-ws/

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -q -y \
    dirmngr \
    gnupg2 \
    lsb-release \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 421C365BD9FF1F717815A3895523BAEEB01FA116

RUN echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -sc` main" > /etc/apt/sources.list.d/ros-latest.list

RUN apt-get update && apt-get install --no-install-recommends -y \
    python-rosdep \
    python-rosinstall \
    python-vcstools \
    && rm -rf /var/lib/apt/lists/*

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt-get update && apt-get install -y \
    ros-kinetic-desktop-full \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init \
    && rosdep update

RUN useradd -ms /bin/bash $USERNAME

RUN echo "${USERNAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

USER $USERNAME

RUN source /opt/ros/kinetic/setup.sh && \
    mkdir -p $WORKSPACE/src && \
    cd $WORKSPACE/src && \
    git clone https://github.com/karlpauwels/simtrack.git && \
    cd $WORKSPACE && \
    if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then sudo rosdep init; fi && \
    rosdep update && \
    sudo apt-get update && \
    rosdep install --from-paths src --ignore-src -y -r && \
    sudo rm -rf /var/lib/apt/lists/*

WORKDIR $WORKSPACE
