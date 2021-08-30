FROM python:3.8
WORKDIR /code
COPY cloud_requirements.txt .
RUN pip install -r cloud_requirements.txt
COPY app.py .
COPY bert_ner_model.py .
ADD weights ./weights
ADD templates ./templates
ADD static ./static
ENTRYPOINT ["python", "app.py"]