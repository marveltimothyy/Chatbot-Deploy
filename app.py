import os
from symbol import eval_input
import torch
import Model
import Voc
import re
import unicodedata
import torch.nn as nn
from flask import Flask, render_template, request

device = torch.device('cpu')
cwd = os.getcwd()
file = os.listdir(cwd)
corpus_name = "cornell movie-dialogs corpus"
# corpus = os.path.join("content/drive/MyDrive/Skripsi/Code/data", corpus_name)
# a = os.path.join("/save/512-1028-25-0.001/15_checkpoint.tar")

voc = Voc.Voc(corpus_name)

model_name = 'cb_model'
attn_model = 'dot'
max_length = 10
hidden_size = 2048
batch_size = 128
learning_rate = 0.0001
epoch = 15
loadFilename = 'Save/'+'{}-{}-{}-{}'.format(
    hidden_size, batch_size, epoch, learning_rate)+'/15_checkpoint.tar'

hidden_size2 = 512
batch_size2 = 256
learning_rate2 = 0.0001
epoch2 = 15
loadFilename2 = 'Save/'+'{}-{}-{}-{}'.format(
    hidden_size2, batch_size2, epoch2, learning_rate2)+'/15_checkpoint.tar'
# print('Loadfilename', loadFilename)

checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
checkpoint2 = torch.load(loadFilename2, map_location=torch.device('cpu'))

voc = Voc.Voc(corpus_name)
voc.__dict__ = checkpoint['voc_dict']

model1 = Model.Model(checkpoint, loadFilename, hidden_size, batch_size,
                     learning_rate, device, max_length, voc)
model2 = Model.Model(checkpoint2, loadFilename2, hidden_size2, batch_size2,
                     learning_rate2, device, max_length, voc)
# model = Model.Model(checkpoint, loadFilename, hidden_size, batch_size,
#                     learning_rate, device, max_length, voc)


# create and configure the app
app = Flask(__name__, instance_relative_config=True)
app.secret_key = 'bangkit'


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get", methods=["GET"])
def get_response():
    userText = str(request.args.get('msg'))

    return str(model1.response(userText))


@app.route("/2")
def index2():
    return render_template("index2.html")


@app.route("/get2", methods=["GET"])
def get_response2():
    userText = str(request.args.get('msg'))
    return str(model2.response(userText))
