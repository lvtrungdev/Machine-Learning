FROM python:3.12
LABEL authors="lvtrung"

WORKDIR                 /src

COPY diabetes.pkl       /src/diabetes.pkl
COPY server.py          /src/server.py
COPY requirements.txt   /src/requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8888

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8888"]