FROM b.gcr.io/tensorflow/tensorflow

RUN pip install sklearn

EXPOSE 5000:5000

RUN mkdir /code
WORKDIR "/code"

COPY ./requirements.txt /code/requirements.txt
RUN pip install -r /code/requirements.txt

COPY ./ /code

RUN sh get_data.sh
RUN python nah.py train

CMD python nah.py launch
