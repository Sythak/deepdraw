FROM --platform=linux/amd64 tensorflow/tensorflow:2.10.0
COPY deep_draw deep_draw
COPY requirements_prod.txt requirements_prod.txt
COPY setup.py setup.py
RUN apt update && apt install -y libcairo2-dev
RUN pip install --upgrade pip
RUN pip install .
CMD uvicorn deep_draw.fast_api:app --host 0.0.0.0 --port $PORT
