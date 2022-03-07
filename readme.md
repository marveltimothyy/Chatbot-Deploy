## Description
This is my chatbot repository for deployment purpose. For a simple tutorial about build generative chatbot using pytorch you can visit <a href="https://pytorch.org/tutorials/beginner/chatbot_tutorial.html">Pytorch Chatbot Tutorial</a>

## Main Library 📚
• <a href="https://pytorch.org/">Pytorch</a> 
<br>
• <a href="https://flask.palletsprojects.com/en/2.0.x/" >Flask</a>
## Model 🤖
My chatbot using sequence-to-sequence architecture wich have Encoder and Decoder in it. For each Encoder and Decoder i'm using Bidirectional Gated Recurrent Unit imported from pytroch.
In additon to improve chatbot performance i'm using <a href="https://arxiv.org/abs/1508.04025">Luong Attention Mechanism</a>. <br> <br>
There is two different model that i'm deploying, the first one is the model who have good performance in bleu score, the second model is the best model in hyperparameter tunning process

## Deploy Services 🚀
I'm using droplet from <a href="https://www.digitalocean.com/">Digital Ocean</a> with a little addition in memory. You can access my deployed model 
through this link : <br>
• <a href='http://206.189.87.31:5000/'>First Model<a> <br>
• <a href='http://206.189.87.31:5000/2'>Second Model<a>
