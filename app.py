from flask import Flask, render_template, url_for, request
import json
from imageio import imread
from fastai.vision import *
import urllib.request
from PIL import Image

app = Flask(__name__)


@app.route('/')
def entry_page():
    # Jinja template of the webpage
    return render_template('index.html')


@app.route('/predict_object/', methods=['GET', 'POST'])
def render_message():
    # Loading CNN model
    # saved_model = 'saved_models/tuned_model_fin.h5'
    # saved_model = 'saved_models/tuned_model_fin.h5'
    path = Path('data')
    classes = ['non_smoker', 'smoker']
    data2 = ImageDataBunch.single_from_classes(path, classes, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)


    # model = load_model(saved_model)
    learn = create_cnn(data2, models.alexnet)
    learn.load('alexnet')

    try:
        # Get image URL as input
        image_url = request.form['image_url']
        img_data = requests.get(image_url).content
        with open('img.jpg', 'wb') as handler:
            handler.write(img_data)
        img = open_image('img.jpg')

        '''
        #Call classify function to predict the image class using the loaded CNN model
        final,pred_class = classify(x, model)
        print(pred_class)
        print(final)
        '''

        pred_class, pred_idx, final = learn.predict(img)

        print(pred_class)
        print(final)

        # Store model prediction results to pass to the web page
        message = "Model prediction: {}".format(pred_class)
        print('Python module executed successfully')

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
    app.run(debug=True)


#{{ data.reset_index(drop = True).to_html(classes="table table-striped") | safe}}
