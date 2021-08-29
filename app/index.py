from tkinter import *
from PIL import Image 
import PIL
from PIL import ImageTk, Image
from tkinter import filedialog
from keras import models
from tensorflow import keras
import numpy as np


root = Tk()
root.geometry("550x300+300+150")
root.resizable(width=True, height=True)

model = models.load_model('covid_19_adamax.h5')

def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename
def open_img():
    x = openfn()
    img = Image.open(x)
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image=img)
    panel.image = img
    panel.pack()
    predic(x)
    

btn = Button(root, text='Abrir imagen', command=open_img).pack()



def predic(predi):
    image_size = (150, 150)
    imgx = keras.preprocessing.image.load_img(predi, target_size=image_size)
    x = keras.preprocessing.image.img_to_array(imgx)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)
    score = predictions[0]
    panel = Label(root, text="Esta imagen tiene %.2f por ciento de COVID-19  y %.2f por ciento Normal." % (100 * (1 - score), 100 * score))
    panel.pack()
    return FALSE

root.mainloop()