FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
# FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04
LABEL description="warpdrive-environment"

RUN apt-get update && yes|apt-get upgrade && apt-get -qq install build-essential
RUN apt-get install -y --no-install-recommends apt-utils
RUN apt-get install -y curl
RUN apt-get install -y wget
RUN apt-get install -y sudo
WORKDIR /home/
RUN chmod a+rwx /home/

# Install miniconda to 
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /home/miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/home/miniconda/bin:${PATH}
RUN conda update -y conda

# Python packages from conda
RUN conda install -c anaconda -y python=3.7
RUN conda install -c anaconda -y pip
RUN pip install ai-economist
RUN pip install gym==0.21
RUN pip install tensorflow==1.14
RUN pip install "ray[rllib]==0.8.4"
RUN pip install rl-warp-drive==1.2.1
RUN apt-get update && apt-get install -y git
ARG commit
# RUN echo "$commit" && git clone https://github.com/mhelabd/ai-ethicist.git
RUN echo "$commit"
ADD . /home/ai-ethicist
# RUN cd ai-ethicist && git checkout $commit
RUN echo "PWD is: $PWD"
CMD [ "python", "./ai-ethicist/ai_economist/training/training_script.py", "--env", "./ai-ethicist/ai_economist/training/run_configs/moral_environment.yaml " ] 
