# Face Embedding Association Test (FEAT)

This repository contains a data and code for the paper **Measuring Embedded Human-like Biases in Face Recognition Models**, published at AAAI2022 Workshop on Artificial Intelligence with Biased or Scarce Data [AIBSD](https://aibsdworkshop.github.io/2022/index.html). We introduced Face Embedding Association Test (FEAT), which is an extension of Word Embedding Association Test [WEAT](https://arxiv.org/pdf/1608.07187.pdf?ref=hackernoon.com) for face recognition models. 


Data
-------------
Our dataset is in ``data`` folder. We employed dataset from [this repo](https://github.com/W4ngatang/sent-bias) and additionally collected dataset to test a wide range of social biases by using google search. 



Setup
-------------
### Experimental requirements
* Python == 3.6
* TensorFlow == 2.4.1
* Keras == 2.4.0

### Dataset
``data`` folder consists of ``targets`` and ``attributes``. We employ target images of all races from [UTKFace](https://susanqq.github.io/UTKFace/). For attributes images of European American and African American, we use images sourced from [Google Image dataset](https://github.com/candacelax/bias-in-vision-and-language/tree/703f559b1d81d51817d6fb7251b901efc28505b6/data/google-images). 

You can also crawl additional attributes dataset using  ``attributes/crawl.py``. To get google image dataset, we use [google_images_download](https://pypi.org/project/google_images_download/) package. Please install the required package.
```
pip install google_images_download
```

### Download pre-trained models
Download the pretrained models for:
* VGGFace

> For VGGFace, we use [keras-vggface](https://pypi.org/project/keras-vggface/) package. Please install the required package.
```
pip install keras-vggface
```


* [FaceNet](https://drive.google.com/file/d/1971Xk5RwedbudGgTIrGAL4F7Aifu7id1/view)
* [Deepface](https://github.com/swghosh/DeepFace/releases)
* [DeepID](https://drive.google.com/file/d/1uRLtBCTQQAvHJ_KVrdbRJiCKxU8m5q2J/view)
* [Openface](https://drive.google.com/file/d/1LSe1YCV1x-BfNnfb7DFZTNpv_Q9jITxn/view)
* [Arcface](https://drive.google.com/uc?id=1LVB3CdVejpmGHM28BpqqkbZP5hDEcdZY)

Note that you need to download pretrained model weights for each model at ``models``.


Experiment
-------------
### FEAT
We introduce Face Embedding Association Test (FEAT) by extending the prior works throughout face embeddings. FEAT compares the embeddings of face images, rather than sets of words or sentences, to demonstrate race and gender. A detailed explanation of FEAT is provided in our paper "Measuring Embedded Human-like Biases in Face Recognition Models". 

### Details
After downloading data and pretrained models, run main.py.
```
python test/run_test.py
```

dfkdlfkdflsd

