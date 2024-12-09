import json
import warnings
from io import BytesIO

import dotenv
import keras
import numpy as np
from PIL import Image
from keras.src.utils import img_to_array

from constants import BASE_DIR

dotenv.load_dotenv()

warnings.simplefilter(action='ignore', category=FutureWarning)

model = keras.saving.load_model(BASE_DIR / 'ml_data/mobilenet.keras')


def get_metrics():
    metrics = json.load(open(BASE_DIR / 'ml_data/metrics.txt'))
    return metrics


def normalize_image(file, neural_network='mlp'):
    img = Image.open(BytesIO(file)).resize((28, 28)).convert('L')

    x = 255 - img_to_array(img)

    x = x.reshape(1, 784) / 255 \
        if neural_network == 'mlp' \
        else np.expand_dims(x, axis=0)

    return x


def image_from_array(obj) -> bytes:
    img = Image.fromarray(obj)
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()
