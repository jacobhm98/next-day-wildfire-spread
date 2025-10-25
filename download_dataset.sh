#!/bin/bash

mkdir data
curl -L -o data/next-day-wildfire-spread.zip\
  https://www.kaggle.com/api/v1/datasets/download/fantineh/next-day-wildfire-spread

unzip data/next-day-wildfire-spread.zip -d data/
