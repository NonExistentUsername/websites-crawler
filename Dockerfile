FROM nvidia/cuda:12.0.1-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y openssh-server
RUN apt-get update && apt-get install sudo
RUN useradd -rm -d /home/sshuser -s /bin/bash -g root -G sudo -u 1000 sshuser
RUN echo 'sshuser:BL93h406yokEs7' | chpasswd
RUN mkdir /var/run/sshd
RUN sudo apt-get install parted -y
RUN sysctl -w net.ipv6.conf.all.disable_ipv6=1 && sysctl -w net.ipv6.conf.default.disable_ipv6=1 && sysctl -w net.ipv6.conf.lo.disable_ipv6=1
EXPOSE 22
RUN apt install software-properties-common -y 
RUN add-apt-repository ppa:deadsnakes/ppa 
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN apt install default-jre
CMD ["/usr/sbin/sshd", "-D"]