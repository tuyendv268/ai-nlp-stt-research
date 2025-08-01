FROM nvcr.io/nvidia/kaldi:22.10-py3

COPY tools/srilm.tar.gz /opt/kaldi/tools
COPY tools/install_liblbfgs.sh /opt/kaldi/tools/extras
COPY tools/install_srilm.sh /opt/kaldi/tools/extras

RUN cd /opt/kaldi/tools && bash install_srilm.sh
