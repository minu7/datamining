FROM python:3

RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    python-dev

ADD . /home

WORKDIR /home

RUN pip install nltk
RUN pip install pandas
RUN pip install numpy
RUN pip install xlrd
RUN pip install Flask
RUN pip install pymongo
RUN pip install scikit-learn
RUN pip install flask-cors
RUN pip install GetOldTweets3
RUN python download.py
