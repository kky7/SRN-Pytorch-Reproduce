From pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel
MAINTAINER ky

RUN apt-get update
RUN apt-get install -y wget
RUN apt-get install -y tmux
RUN apt-get install -y reptyr
RUN pip install tensorboardX 
RUN pip install scipy pillow
RUN pip install opencv-contrib-python
RUN pip install torchvision
RUN pip install scikit-image

WORKDIR /root/