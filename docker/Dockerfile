FROM python:3.9.20-bookworm

RUN apt-get update && apt-get install -y cmake
RUN git clone https://github.com/josephzhong/kick2goal.git
RUN cd kick2goal \
    && python3 -m pip install -r requirements.txt
RUN python3 -m pip install "stable-baselines3[extra]"
RUN mkdir models
