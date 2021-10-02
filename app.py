from flask import Flask, render_template, request
from fastai.vision import *

app = Flask(__name__)

@app.route('/')
def entry_page():
    # Jinja template of the webpage
    return render_template('index.html')


@app.route('/predict_object/', methods=['GET', 'POST'])
def render_message():
    # Loading CNN model
    path = Path('data')
    classes = ['non_smoker', 'smoker']
    data2 = ImageDataBunch.single_from_classes(path, classes, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)

    learn = create_cnn(data2, models.alexnet)
    learn.load('../models/alexnet')

    try:
        # Get image URL as input
        image_url = request.form['image_url']
        img_data = requests.get(image_url).content
        with open('img.jpg', 'wb') as handler:
            handler.write(img_data)
        img = open_image('img.jpg')

        pred_class, pred_idx, final = learn.predict(img)

        print(pred_class)
        print(final)

        # Store model prediction results to pass to the web page
    
        if str(pred_class) == "smoker":
            message = "Model prediction: Smoker ! "
        else:
            message = "Model prediction: Non Smoker ! "
        
        #message = "Model prediction: {}".format(pred_class)
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
    app.run(debug=True,host='0.0.0.0')
