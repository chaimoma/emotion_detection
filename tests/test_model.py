import tensorflow as tf

def test_load_model():

    print("\n--- (pytest) Testing Model Load ---")
    model = tf.keras.models.load_model('saved_model/emotion_model.h5', compile=False)
    

    assert model is not None, "Failed to load the model"