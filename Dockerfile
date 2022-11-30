  FROM tensorflow/tensorflow:2.10.0
  WORKDIR /prod
  COPY deep_draw deep_draw
  COPY requirements.txt requirements.txt
  COPY .env .env
  COPY setup.py setup.py
  RUN pip install --upgrade pip
  RUN pip install -r requirements.txt
  CMD uvicorn deep_draw.fast_api:app --host 0.0.0.0 --port $PORT
