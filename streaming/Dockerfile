FROM python:2.7

RUN mkdir /code
WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install -r /code/requirements.txt

COPY ./ /code

CMD python streamer.py
