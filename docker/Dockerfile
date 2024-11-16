FROM python:3.9.20

RUN git clone https://github.com/josephzhong/kick2goal.git
RUN pip3 install --upgrade pip
RUN cd kick2goal \
    && pip3 install -r requirements.txt \
    && mkdir models
RUN apt-get update && apt-get install -y cmake
RUN pip3 install stable-baselines3[extra]
