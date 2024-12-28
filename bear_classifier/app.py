import gradio as gr
from fastai.vision.all import *
import skimage

learn = load_learner('export.pkl')

labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

title = "Bear Classifier"
description = "A Bear Classifier is trained on GGD search results with fastai. Created as a demo for Gradio and HuggingFace Spaces."
article="<p style='text-align: center'><a href='https://github.com/qasimkhan/Practical-Deep-Learning-for-Coders' target='_blank'>Code Repo</a></p>"
examples = ['grizzly.jpg']

gr.Interface(fn=predict,
             inputs=gr.components.Image(),
             outputs=gr.components.Label(num_top_classes=3),
             title=title,
             description=description,
             article=article,
             examples=examples).launch()
