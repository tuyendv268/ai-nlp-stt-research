FROM python:3.11

USER root

# RUN apt-get update

RUN pip install --upgrade pip

RUN python -m pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[all]
RUN pip install torchaudio==2.7.1
RUN python -m pip install git+https://github.com/NVIDIA/NeMo-text-processing.git@main#egg=nemo_text_processing

ADD requirements.txt /requirements.txt
RUN pip install -r requirements.txt