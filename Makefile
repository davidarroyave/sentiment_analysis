#Makefile for sentiment analysis project

MLFLOW_PORT = 5000

download:
	python src/data/download_data.py

preprocess:
	python src/data/preprocess_data.py

#train:
#	python src/data/train.py

evaluate:
	python src/data/evaluate_model.py

main:
	python main.py

mlflow:
	mlflow ui --backend-store-uri file:./mlruns --port $(MLFLOW_PORT)
