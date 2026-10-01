FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

#Nvidia Public GPG Key
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub

RUN apt update && apt install -y wget unzip curl bzip2 git
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda config --system --set auto_update_conda false && \
    conda config --system --set show_channel_urls true && \
    conda config --system --remove-key channels

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN mkdir /workspace/ && cd /workspace/ && git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git && cd pytorch-CycleGAN-and-pix2pix && pip install -r requirements.txt 2>/dev/null || pip install scipy opencv-python dominate visdom wandb

WORKDIR /workspace/pytorch-CycleGAN-and-pix2pix
