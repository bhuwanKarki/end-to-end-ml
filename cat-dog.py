from numpy.lib.financial import ipmt
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import streamlit as st
from PIL import Image
import numpy as np

st.title("CAT-DOG prediction")
st.write("select the image to predict eg:")
st.image("https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.hdwallpaper.nu%2Fwp-content%2Fuploads%2F2015%2F02%2FFunny-Cat-Hidden.jpg&f=1&nofb=1",width=200)


@st.cache(allow_output_mutation=True)
def prediction(image):
    model=keras.models.load_model("data/train/model")
    image_array=np.array(image)
    image=tf.expand_dims(tf.image.resize(image_array,(160,160)),axis=0)
    pred=model.predict(image)
    return pred
    
upload_img=st.file_uploader('upload an image of cat or dog',type="jpg")
if upload_img is not None:
    image=Image.open(upload_img)
    st.write("")
    st.write("classifying the image....")
    st.image(upload_img,width=200)
    predict=prediction(image)
    st.markdown("the predicted image is:")
    if predict[0][0]>0.5:
        st.write("dog")
    else:
        st.write("cat")