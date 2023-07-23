import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify

app = Flask(__name__)

# Define the image size expected by the models
img_height, img_width = 224, 224

# Define the class names
class_names = ['Very mild Dementia', 'Non Demented', 'Moderate Dementia', 'Mild Dementia']

# preprocess the image for each model
def preprocess_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_height, img_width])
    img = img / 255.0  # Normalize pixel values between 0 and 1
    return img

@app.route('/predict', methods=['POST'])
def predict_image():
    if request.method == 'POST':
        
        files = request.files.getlist('file')
        predictions = []
        
        for idx, file in enumerate(files):
            if file.filename == '':
                return jsonify({'error': f'File {idx+1}: No file selected'})

            try:
                # Save the uploaded image to a temporary location
                image_path = f'temp_{idx}.jpg'
                file.save(image_path)
                
                # tf.random.set_seed(42)

                # Preprocess the image for each model
                with tf.device('/GPU:0'):  # Specify the GPU device to use
                    vgg_model = load_model(vgg_model_path)
                    vgg_img = preprocess_image(image_path)
                    vgg_img_array = tf.expand_dims(vgg_img, 0)
                    vgg_predictions = vgg_model.predict(vgg_img_array)
                    vgg_class = class_names[np.argmax(vgg_predictions)]
                    print('vgg:', vgg_predictions)
                    del vgg_model  # Delete the model to free up GPU memory
                
                with tf.device('/GPU:0'):
                    resnet_model = load_model(resnet_model_path)
                    resnet_img = preprocess_image(image_path)
                    resnet_img_array = tf.expand_dims(resnet_img, 0)
                    resnet_predictions = resnet_model.predict(resnet_img_array)
                    resnet_class = class_names[np.argmax(resnet_predictions)]
                    print('resnet:', resnet_predictions)
                    del resnet_model
                
                with tf.device('/GPU:0'):
                    inceptionres_model = load_model(inceptionres_model_path)
                    inception_img = preprocess_image(image_path)
                    inception_img_array = tf.expand_dims(inception_img, 0)
                    inception_predictions = inceptionres_model.predict(inception_img_array)
                    inceptionresnet_class = class_names[np.argmax(inception_predictions)]
                    print('inceptionres', inception_predictions)
                    del inceptionres_model
                
                 # Append the prediction result
                predictions.append({
                    'file': file.filename,
                    'vgg_prediction': vgg_class,
                    'resnet_prediction': resnet_class,
                    'inceptionresnet_prediction': inceptionresnet_class
                })

            except Exception as e:
                return jsonify({'error': f'File {idx+1}: {str(e)}'})
            finally:
                # Delete the temporary image file
                if os.path.exists(image_path):
                    os.remove(image_path)
        
    return jsonify(predictions)        
    
if __name__ == '__main__':
    
    # Load the trained models here before starting the Flask app
    vgg_model_path = 'alzheimers_VGG16.h5'
    resnet_model_path = 'alzheimers_ResNet50.h5'
    inceptionres_model_path = 'alzheimers_InceptionResNetV2.h5'

    app.run(debug=True, port=3030)