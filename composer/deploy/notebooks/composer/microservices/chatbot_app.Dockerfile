FROM python:3.11-slim

RUN apt-get -y update && \
    apt-get -y install python3-pip

# pip install the other python requirements in the container
WORKDIR .
COPY chatbot_requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# copy the src code into the directory in the container
COPY chatbot* /app/
WORKDIR /app

ENV PYTHONUNBUFFERED 1

ENTRYPOINT python chatbot_app.py
