import tensorflow as tf
from train import get_siamese_model

model = get_siamese_model((105, 105, 1))
model.summary()
model.load_weights('./finalModel/oneshot_weights.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

f = open('./finalModel/siamese.tflite','wb')
f.write(tflite_model)
f.close()
