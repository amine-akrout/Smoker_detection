from flask import Flask, render_template, request
import pandas as pd
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import requests


CLASSES = ['Non Smoker', 'Smoker']
SIZE=150
# MODEL_URI='http://localhost:8501/v1/models/smoker_detector:predict'
# MODEL_URI='http://172.17.0.2:8501/v1/models/smoker_detector:predict'
MODEL_URI='http://tf_serving:8501/v1/models/smoker_detector:predict'
def preprocess_img(path):
    img = image.load_img(path, target_size=(SIZE, SIZE))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = np.vstack([img])
    return img

app = Flask(__name__)

@app.route('/')
def entry_page():
    # Jinja template of the webpage
    return render_template('index.html')


@app.route('/predict_object/', methods=['GET', 'POST'])
def render_message():
    try:
        # Get image URL as input
        image_url = request.form['image_url']
        img_data = requests.get(image_url).content
        with open('img.jpg', 'wb') as handler:
            handler.write(img_data)
        img = preprocess_img('img.jpg')
        data = json.dumps({"signature_name": "serving_default", "instances": img.tolist()})
        headers = {"content-type": "application/json"}
        json_response = requests.post(MODEL_URI, data=data, headers=headers)
        predictions = json.loads(json_response.text)['predictions']
        pred_class = CLASSES[int(predictions[0][0])]
        final = pred_class

        if str(pred_class) == "Smoker":
            message = "Model prediction: Smoker ! "
        else:
            message = "Model prediction: Non Smoker ! "
        
        print('Python module executed successfully')
        print (str(pred_class))

    except Exception as e:
        # Store error to pass to the web page
        message = "Error encountered. Try another image. ErrorClass: {}, Argument: {} and Traceback details are: {}".format(
            e.__class__, e.args, e.__doc__)
        final = pd.DataFrame({'A': ['Error'], 'B': [0]})

    # Return the model results to the web page
    return render_template('index.html',
                           message=message,
                           data=final,
                           image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True,host= '0.0.0.0', port=8080)
