# Deploying CNN based Tensorflow model using Tensorflow Serving, Flask, Docker and Docker compose

Deploying a Tensorflow InceptionV3 model with Flask for smoker detection in a contenairized environnement


## Project Organization

```
│───app
│   ├───app.py
│   ├───Dockerfile 
│   ├───requirements.txt
│   ├───static
│   └───templates
│
├───data
│   ├───models
│   ├───non_smoker
│   └───smoker
├───data_tf
│   └───smoker-v-non_smoker
│       ├───testing
│       │   ├───non_smoker
│       │   └───smoker
│       └───training
│           ├───non_smoker
│           └───smoker
├───demo
│       app_demo.gif
│       demo_app.gif
│       img.jpg
│
└───smoker_detection
    │   saved_model.pb
    │
    ├───assets
    └───variables
            variables.data-00000-of-00001
            variables.index
```

## Only 3 steps to run !

### 1. Clone the repository
`git clone https://github.com/amine-akrout/Smoker_detection.git `

`cd Smoker_detection`

### 2. Build and run the Docker Stack (TensorFlow serving and Web App)
` docker-compose build .`

` docker-compose up`

### 3. Open the Web App and try it out !
The App should be running on ` http://localhost:5000`


## What the app looks like

![demo](https://github.com/amine-akrout/Smoker_detection/blob/master/demo/app_demo.gif)



#### References:
- [Convolutional Neural Networks in TensorFlow](https://www.example.com) 

- https://github.com/rexsimiloluwah/fastapi-tensorflow-serving-old

- https://github.com/huyhoang17/tensorflow-serving-docker