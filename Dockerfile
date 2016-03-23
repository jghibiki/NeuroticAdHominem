FROM b.gcr.io/tensorflow/tensorflow

RUN pip install sklearn

RUN mkdir /code

COPY ./ /code

EXPOSE 8000

WORKDIR "/code"

CMD ["/bin/bash"]
