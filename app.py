from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
model = load_model("maize_disease_model.h5")
class_names = ['Blight', 'Common Rust', 'Gray Leaf Spot', 'Healthy']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    filename = None

    if request.method == 'POST':
        f = request.files['file']
        filename = f.filename
        filepath = os.path.join('static', filename)
        f.save(filepath)

        img = image.load_img(filepath, target_size=(256, 256))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        pred = model.predict(img_array)
        predicted_class_index = np.argmax(pred, axis=1)[0]
        prediction = class_names[predicted_class_index]

    return render_template('index.html', prediction=prediction, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
