import threading
import numpy
from PIL import Image
from keras import Model

from roop.typing import Frame

PREDICTOR = None
THREAD_LOCK = threading.Lock()
MAX_PROBABILITY = 0.85

USE_OPENNSFW2 = False

def get_predictor() -> Model:
    global PREDICTOR

    with THREAD_LOCK:
        if PREDICTOR is None and USE_OPENNSFW2:
            import opennsfw2
            PREDICTOR = opennsfw2.make_open_nsfw_model()
    return PREDICTOR


def clear_predictor() -> None:
    global PREDICTOR

    PREDICTOR = None


def predict_frame(target_frame: Frame) -> bool:
    if not USE_OPENNSFW2:
        return False

    import opennsfw2
    image = Image.fromarray(target_frame)
    image = opennsfw2.preprocess_image(image, opennsfw2.Preprocessing.YAHOO)
    views = numpy.expand_dims(image, axis=0)
    _, probability = get_predictor().predict(views)[0]
    return probability > MAX_PROBABILITY


def predict_image(target_path: str) -> bool:
    if not USE_OPENNSFW2:
        return False

    import opennsfw2
    return opennsfw2.predict_image(target_path) > MAX_PROBABILITY


def predict_video(target_path: str) -> bool:
    if not USE_OPENNSFW2:
        return False

    import opennsfw2
    _, probabilities = opennsfw2.predict_video_frames(video_path=target_path, frame_interval=100)
    return any(probability > MAX_PROBABILITY for probability in probabilities)
