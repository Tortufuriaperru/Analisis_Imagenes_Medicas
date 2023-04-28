import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf

from tempfile import NamedTemporaryFile
from tensorflow.keras.preprocessing import image

#st.set_option('deprecation.showfileUploaderEncoding', False)
#@st.cache(allow_output_mutation=True)


def loading_model():
  fp = "models/model3.h5"
  model_loader = load_model(fp)
  return model_loader

cnn = loading_model()
st.write("""
# Clasificación de radiografias de torax
""")



temp = st.file_uploader("Cargue una imagen: ", type=["png", "jpg", "jpeg"])
#temp = temp.decode()

buffer = temp
temp_file = NamedTemporaryFile(delete=False)
if buffer:
    temp_file.write(buffer.getvalue())
    st.write(image.load_img(temp_file.name))


if buffer is None:
     st.text("Oops! Esto no es una imagen, carga de nuevo.")

else:
    img = image.load_img(temp_file.name, target_size=(299, 299))

    # Preprocessing the image
    pp_img = image.img_to_array(img)
    pp_img = pp_img/299
    pp_img = np.expand_dims(pp_img, axis=0)

    #predict
    prediction= cnn.predict(pp_img)

    result = np.argmax(prediction,axis=1)[0]
    accuracy = float(np.max(prediction,axis=1)[0])
    label_dict={0:'Covid19 positivo', 1:'Opacidad pulmonar', 2: 'Normal', 3:'Neumonia viral'}
    label= label_dict[result]

    response = {'prediction': {'result': label,'accuracy': accuracy}}
    st.success("Con una precisión del {0} la imagen seleccionada es {1}".format(response["prediction"]["accuracy"],response["prediction"]["result"]))

    image = Image.open(temp)
    st.image(image,use_column_width=True)
