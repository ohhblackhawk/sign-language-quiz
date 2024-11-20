FROM python:3.10-slim

WORKDIR /app 

# Install dependencies
RUN apt update
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 libgl1-mesa-glx


COPY requirements.txt /app/requirements.txt  
RUN pip install -r /app/requirements.txt


COPY static /app/static
COPY templates /app/templates
COPY app.py /app/app.py
COPY cnn_model.h5 /app/cnn_model.h5

RUN ls -l /app/static/images

EXPOSE 5000

CMD ["python", "/app/app.py"]