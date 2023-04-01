FROM python:3.9-slim

# Install libraries
COPY ./requirements.txt ./
RUN pip install -r requirements.txt && \
    rm ./requirements.txt

# Setup container directories
RUN mkdir /web

# Copy local code to the container
COPY ./web /web

# launch server with gunicorn
WORKDIR /web
EXPOSE 5000
CMD ["gunicorn", "main:app", "--timeout=0", "--preload", \
     "--workers=1", "--threads=4", "--bind=0.0.0.0:5000"]