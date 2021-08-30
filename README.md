# Example Of Fine-Tuning BERT For Named-Entity Recognition Task And Preparing For Cloud Deployment Using Flask, React, And Docker

#### This repository contains helpful code snippets and configuration for fine-tuning BERT for the downstream task, wrapping model using Flask, and deploying as a Docker container

<img src="https://github.com/dredwardhyde/bert-ner-fine-tuning-cloud-deployment/blob/main/result.png" width="862"/>  

### How to use this repository:
#### - Install requirements
```sh
pip install -r requirements.txt
```
#### - Fine-tune BERT
```sh
python bert_ner_fine_tuning.py
```
#### - Build React UI
```sh
cd react
npm install
npm run build
```
#### - Build Docker Image
```sh
docker build -t bert-ner-fine-tuning-cloud-deployment .
```
#### - Deploy Docket Image
```sh
docker-compose -f docker-compose.yml up -d
```