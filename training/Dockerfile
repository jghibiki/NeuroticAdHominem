FROM b.gcr.io/tensorflow/tensorflow

RUN pip install sklearn

RUN mkdir /code
WORKDIR "/code"

COPY ./requirements.txt /code/requirements.txt
RUN pip install -r /code/requirements.txt

COPY ./ /code

RUN sh get_data.sh

RUN python pretrain.py

CMD python start.py
