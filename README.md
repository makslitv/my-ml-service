# ML-Service for Number Classification
## Project Description

This project implements a machine learning service for classifying handwritten digits from the MNIST dataset. The service takes input as an array of length 784 (28x28 grayscale pixels) and returns the predicted class (0 to 9).

my-ml-service/
```               
├── models/            
├── scripts/            
│   ├── train.py        
├── service/            
│   ├── app.py          
│   ├── requirements.txt
│   ├── Dockerfile      
├── docker-compose.yml  
└── README.md           
```

## Installation and launch
Clone repository:

bash
   ```
   git clone <link to this repo>
   cd my-ml-service
```


## Create an environment and install dependencies:

bash
```
conda create -n torch_env python=3.9
conda activate torch_env
pip install -r service/requirements.txt
```

## Train the model:

bash
```
python scripts/train.py
```


## Start the service:

bash
```
python service/app.py
```

## Send request: In another terminal, run:

bash
```
curl -X POST -H "Content-Type: application/json" \
-d "$(python3 -c 'import json; import random; print(json.dumps({"image": [random.random() for _ in range(784)]}))')" \
http://127.0.0.1:8080/predict
```

## Use with Docker
Build the image and run the container:

```
css
Copy code
docker-compose up --build
```

Send request:

```
curl -X POST -H "Content-Type: application/json" \
-d "$(python3 -c 'import json; import random; print(json.dumps({"image": [random.random() for _ in range(784)]}))')" \
http://127.0.0.1:8080/predict
```

## Model information
A pre-trained ResNet18 model from PyTorch is used.
The model is trained to classify 10 classes (numbers 0 to 9).
Logging of parameters, metrics and model is done through MLflow.


Author
M. Litvinov
MDS HSE
