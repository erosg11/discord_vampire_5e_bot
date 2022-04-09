FROM python:3.9.12-slim-buster

WORKDIR /home/bot/

COPY requirements.txt requirements.txt

RUN apt update && apt install -y ffmpeg libsm6 libxext6 && pip install -r requirements.txt

COPY d10_faces.png d10_faces.png

COPY main.py main.py


CMD ["python", "main.py"]