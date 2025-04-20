from joblib import load
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
import cv2
import os
import matplotlib.pyplot as plt

from tensorflow.keras.saving import register_keras_serializable
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import tensorflow as tf

@keras.saving.register_keras_serializable()
class AnomalyDetector(Model):
  def __init__(self, input_shape):
    super(AnomalyDetector, self).__init__()
    self.input_shape_ = input_shape  # Store input_shape as an instance variable
    self.encoder = tf.keras.Sequential([
      layers.Dense(32, activation="relu"),
      layers.Dense(16, activation="relu"),
      layers.Dense(8, activation="relu")])

    self.decoder = tf.keras.Sequential([
      layers.Dense(16, activation="relu"),
      layers.Dense(32, activation="relu"),
      layers.Dense(input_shape, activation="sigmoid")])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

  def get_config(self):
      """
      Save the configuration of the model.
      """
      return {"input_shape": self.input_shape_}

  @classmethod
  def from_config(cls, config):
      """
      Recreate the model from its configuration.
      """
      return cls(input_shape=config["input_shape"])

import tensorflow as tf

# Load your models
ss = load('models/ss_model (1).pkl')
pca = load('models/pca_model (1).pkl')
# oc_svm_clf = load('oc_svm_model.pkl')
min_val = tf.constant(-7.987133979797363, dtype=tf.float64)
max_val = tf.constant(9.166524887084961, dtype=tf.float64)
autoencoder2 = tf.keras.models.load_model('models/autoencoder_standard_scaled_5625.keras')
segment_model = tf.keras.models.load_model('models/segment_mode.keras')
image_size = 224
resnet_model = ResNet50(input_shape=(image_size, image_size, 3), include_top=False, pooling='avg')  # Since top layer is the fc layer used for predictions

import numpy as np
threshold = 0.06962311267852783

def predict(model, data, threshold):
  reconstructions = model(data)
  loss = tf.keras.losses.mae(reconstructions, data)
  print(loss)
  return tf.math.less(loss, threshold)

import cv2
import numpy as np
import pandas as pd

def safe_destroy(window_name):
    try:
        cv2.destroyWindow(window_name)
    except:
        pass  # Ignore the error if the window doesn't exist

def make_prediction(video_path, max_duration):
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'H264')

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Input FPS: {fps}")
    max_frames = int(fps * max_duration)
    frame_count = 0

    # ✅ new: dynamically grab input’s FPS and frame size
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        "temp/inference_output.mp4",
        fourcc,
        fps,
        (width, height)
    )

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    count = 0

    prev = False

    # Read and display the video frame by frame
    while True:
        ret, frame_og = cap.read()

        frame_count += 1
        if frame_count > max_frames:
            print("10-second frame limit reached.")
            break

        # Break the loop if the video ends
        if not ret:
            print("End of video.")
            break

        frame = cv2.resize(frame_og, (224, 224))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = np.array(frame)

        preprocessed_frame = preprocess_input(frame)

        input_frame = np.expand_dims(preprocessed_frame, axis=0)

        features = resnet_model.predict(input_frame)

        val1 = ss.transform(features)

        val2 = pca.transform(val1)

        val2_df = pd.DataFrame(val2)
        # print(val2_df.head())
        # break
        check = val2_df.values

        new_min_val = tf.reduce_min(check)
        new_max_val = tf.reduce_max(check)

        check = (check - min_val) / (max_val - min_val)
        # test_data = (test_data - min_val) / (max_val - min_val)

        check = tf.cast(check, tf.float32)
        # test_data = tf.cast(test_data, tf.float32)

        # print(val2)

        preds = predict(autoencoder2, check, threshold)
        if prev == True and preds[0] == False:
            count += 1
        if count == 5:
            count = 0
            prev = False
        # print(preds.numpy())
        # break
        if preds[0] == False and prev == False:
            safe_destroy('No road ahead')
            font = cv2.FONT_HERSHEY_SIMPLEX 
            # Use putText() method for 
            # inserting text on video 
            cv2.putText(frame_og,  
                        'No road detected ahead',
                        (50, 50),  
                        font, 1,  
                        (0, 255, 255),  
                        2,  
                        cv2.LINE_4)
            out.write(frame_og)
            prev = False
        else:
            safe_destroy('No road ahead')
            prev = True
            image = cv2.resize(frame_og, (500,500))
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # print(image.shape)
            tensor_im = tf.convert_to_tensor(image)
            # print(tensor_im.shape)

            input_image = tf.image.resize(tensor_im, (128, 128))
            # print(input_image.shape)

            input_image = tf.cast(input_image, tf.float32) / 255.0
            # print(input_image.shape)

            preds = segment_model.predict(input_image[tf.newaxis, ...])
            pred_mask = tf.math.argmax(preds, axis=-1)
            pred_mask = pred_mask[..., tf.newaxis]
            output_image = tf.squeeze(pred_mask)
            # plt.imshow(output_image)
            output_image_np = (output_image.numpy() * 255).astype('uint8')
            output_frame = cv2.resize(output_image_np, (500,500), interpolation=cv2.INTER_NEAREST)
            final = frame_og
            output_frame_resized = cv2.resize(output_frame, (final.shape[1], final.shape[0]), interpolation=cv2.INTER_NEAREST)

            if len(output_frame_resized.shape) == 2:  # Check if grayscale
                output_frame_rgb = cv2.cvtColor(output_frame_resized, cv2.COLOR_GRAY2BGR)
            else:
                output_frame_rgb = output_frame_resized

            added_image = cv2.addWeighted(final, 0.7, output_frame_rgb, 0.3, 0)
            # break
            # rgb_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)
            out.write(added_image)
            # out = cv2.VideoWriter('let\'s check.avi', fourcc, 20.0, (640,  480))
            # cv2.imshow("Check", frame)
            # Exit the video display on pressing 'q'
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break

    # Release the VideoCapture object and close windows
    out.release()
    cap.release()
    # cv2.destroyAllWindows()
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "temp", "inference_output.mp4"))
    return output_path
