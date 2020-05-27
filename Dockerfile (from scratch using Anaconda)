FROM centos:latest

MAINTAINER Khushali_thakkar <khushali.thakkar9@gmail.com>

#ARG TENSORFLOW_VERSION=0.12.1 
#ARG KERAS_VERSION=1.2.0

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN yum update -y
RUN yum install -y wget
RUN yum install -y curl

#Making a local folder to stroe the model
RUN mkdir /root/model/
VOLUME /root/model/


#Installing anaconda
RUN wget \ 
	https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
	&& mkdir /root/.conda \
	&& bash Miniconda3-latest-Linux-x86_64.sh -b \
	&& rm -f Miniconda3-latest-Linux-x86_64.sh

# Install pip
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
	python get-pip.py && \
	rm -rf get-pip.py

RUN conda --version
RUN conda install numpy -y
#RUN conda install matplotlib -y
RUN conda install tensorflow -y
RUN conda install keras -y
#RUN conda install opencv -y
RUN conda install pillow -y
#RUN conda install smtplib

WORKDIR /root/model/
#Running python application
CMD ["bin/bash"]
CMD ["python3","model.py"]

