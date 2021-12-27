FROM python:3.7-slim

# chainging the work directory
WORKDIR /app

# copy the requirements.txt file to the container
COPY requirements.txt .

# installing requirements   
RUN pip install -r requirements.txt
RUN apt-get update

# copy all the files to the container
COPY . .

# set the access keys for s3
#ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}  # get from the env variables
#ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

# download the data from s3
RUN dvc pull

EXPOSE 8501

# run the pipeline
CMD [ "streamlit", "run","cat-dog.py" ]

