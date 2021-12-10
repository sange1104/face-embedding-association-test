# Face Embedding Association Test (FEAT)

This repository contains a data and code for the paper **Measuring Embedded Human-like Biases in Face Recognition Models**, published at AAAI2022 Workshop on [Artificial Intelligence with Biased or Scarce Data](https://aibsdworkshop.github.io/2022/index.html). We introduced Face Embedding Association Test (FEAT), which is an extension of [Word Embedding Association Test](https://arxiv.org/pdf/1608.07187.pdf?ref=hackernoon.com) for face recognition models. 


Data
-------------
Our dataset is in ``data`` folder. We employed dataset from [this repo](https://github.com/W4ngatang/sent-bias) and additionally collected dataset to test a wide range of social biases by using google search. We only uploaded the dataset that we newly crawled. You can use ``crawl.py`` to crawl additional data. To get google image dataset, we use [google_images_download](https://pypi.org/project/google_images_download/) package. Please install the required package.
```
pip install google_images_download
```   
Below is the data directory structure. 

<pre>/data/attributes  
 ┬  
 ├ [DIR] race 
     └ [DIR] 8 asian attributes
 ├ [DIR] age  
     └ [DIR] 8 attributes
         ├ young
         └ old
 ├ [DIR] intersectional 
     ├ [DIR] competent 
         ├ asian_female
         ├ black_female
         └ white_female
     └ [DIR] incompetent 
         ├ asian_female
         ├ black_female
         └ white_female
</pre>


Experimental Setup
-------------
* Python == 3.6
* TensorFlow == 2.4.1
* Keras == 2.4.0
* dlib
* cv2
* pillow
* [skin](https://github.com/cirbuk/skin-detection)

### Download pre-trained models
Download the pretrained models for:
* VGGFace

> For VGGFace, we use [keras-vggface](https://pypi.org/project/keras-vggface/) package. Please install the required package.
```
pip install keras-vggface
```

You can update the weights of the models by using links below:
* [FaceNet](https://drive.google.com/file/d/1971Xk5RwedbudGgTIrGAL4F7Aifu7id1/view)
* [Deepface](https://github.com/swghosh/DeepFace/releases)
* [DeepID](https://drive.google.com/file/d/1uRLtBCTQQAvHJ_KVrdbRJiCKxU8m5q2J/view)
* [Openface](https://drive.google.com/file/d/1LSe1YCV1x-BfNnfb7DFZTNpv_Q9jITxn/view)
* [Arcface](https://drive.google.com/uc?id=1LVB3CdVejpmGHM28BpqqkbZP5hDEcdZY)


### Experiment 1: FEAT on social biases (race, gender, age, intersectional)

You can run ``run_test.py`` function to run FEAT test on openface, arcface, vggface, deepface, facenet, and deepid models.

``test/run_test.py``

### Experiment 2: Race sensitivity analysis

You can run ``change_total`` function in ``race_transformation/black_to_white.py`` to convert race black to white. 
And ``change_total`` function in ``race_transformation/white_to_black.py`` to convert race white to black. 

