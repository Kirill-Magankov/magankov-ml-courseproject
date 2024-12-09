from io import BytesIO

import numpy as np
import tensorflow
from PIL import Image
from fastapi import APIRouter, File
from keras.src.utils import img_to_array
from typing_extensions import Annotated

import helpers
from constants import CLASS_NAMES, CLASS_DESCRIPTIONS

router = APIRouter()
model = helpers.model


@router.post("/classification")
async def api_classification(image: Annotated[bytes, File()], ):
    img = Image.open(BytesIO(image)).resize((224, 224))

    x = img_to_array(img)
    x = tensorflow.expand_dims(x, 0)
    prediction = model.predict(x, verbose=False)
    class_index = np.argmax(prediction)

    return {
        "msg": {
            "class": CLASS_NAMES[class_index],
            "description": CLASS_DESCRIPTIONS[class_index],
        },
        "probs": str(round(prediction[0][class_index], 3)),  # probability
    }
