# Disaster Message Classifier

This repository contains a model for classifying disaster messages, along with the necessary data used for training and a web application to interact with the trained model.

## Contents

1. [Description](#description)
2. [Data](#data)
3. [Model](#model)
4. [Web Application](#web-application)

## Description

The goal of this project is to classify disaster messages into relevant categories, aiding in effective response during emergencies. The model is trained on a dataset containing labeled disaster messages, and it can categorize new messages into predefined categories such as 'water', 'medical_help', 'food', etc.

## Data

The dataset used for training the model is available in the `data` directory. It includes labeled messages along with their respective categories. The dataset is split into training and testing sets for model evaluation.

## Model

The trained model is saved in `classifier.pkl`. It is a machine learning model capable of classifying text messages into relevant categories. The model is built using [insert machine learning library/framework here] and achieved [insert performance metrics here] on the test set.

## Web Application

The web application allows users to interact with the trained model. To run the web app, ensure you have all dependencies installed by running:

`pip install -r requirements.txt`


Once dependencies are installed, you can start the web application by executing:

`python3 run.py`
