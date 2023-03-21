FROM ubuntu:latest
WORKDIR /app
RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev build-essential hdf5-tools libgl1 libgtk2.0-dev

COPY . /app
RUN pip install -r requirements.txt
EXPOSE 3000

RUN echo "completed!"
CMD ["python3","api.py"]
