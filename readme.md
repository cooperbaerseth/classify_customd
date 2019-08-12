Custom Dataset Image Classification w/ Transfer Learning
========================================================
by Seth Baer
7/10/19

NOTE: This code was developed using python 3.5 and 3.7. There will most likely be issues trying to use python 2.

What does this code do?

This code package allows for an easy way to use transfer learning with a state of the art neural network architecture 
(densenet) on a dataset curated by the user. Once the dataset has been collected and labeled (labeling method outlined 
below), one can use this code to classify new images from the same classes. This code has two different versions: the 
single-label version and the multi-label version. Fair warning, the multi-label classification problem is much harder, 
so it requires much more data to get high accuracy. 

At a high level, the process for using the neural net is as follows:
1) Label a set of images (no less than 20 examples per class)
2) Package the images into the correct folder format (details on this later)
3) Run the training script
4) Gather new unlabeled images
5) Run the inference script 

Labeling a set of images and running the training script:

### Single-Label

In order to label a set of images, organize them into subfolders, where the each subfolder's name is the image class for
all images in that subfolder. For example:

### Labeled_Images
```
|
|-> Tower
|	|
|	|-> image1.jpg
|	|
|	|-> image2.jpg
|	|
|	|-> image3.jpg
|
|-> Road
|	|
|	|-> image4.jpg
|	|
|	|-> image5.jpg
|	|
|	|-> image6.jpg
|
|-> Fence
|	|
|	|-> image7.jpg
|	|
|	|-> image8.jpg
|	|
|	|-> image9.jpg
```

### Multi-Label

In order to give labels to a set of images for training, create a csv file as such:

* each row corresponds to an image in the training set
* the first value of a row is the name of the image (chainlink_fence.jpg)
* the following values are the associated labels (field codes)
* name the file img-labels.csv
* make sure this file is in the same directory as the associated training images


Here is an example of what the contents of img-labels.csv might look like:
```
c-1.JPG,cab
ac-2.JPG,cab,gravel
ac.jpg,roof
access.jpg,roof
cab-1-2-2.JPG,building,cab,gravel
cab-1-2.JPG,building,cab,gravel
cab-1.jpg,cab,gravel,fence
cab-2.jpg,gravel,cab
cab-3+h-frame.jpg,cab,gravel,fence
```

# Training

The training script (train.py) allows for 3 arguments: 'run_name', 'dataset_directory', and 'num_trials'. All
parameters are optional, and if the script is run without any arguments, a default test run will start. The default run
might be a good place to start.

**run_name:**

This will define the name of the current training run. It will be the name of the folder which stores the outputs of
the training script. This is an important parameter, and I suggest naming it something unique and such as date_time. The inference script will need to reference this name in order to load the correct model.

**dataset_directory:**

This should be the path to the labeled dataset. The training script will load the labeled data from this directory.
Again, the folder MUST be the correct format for the training to run correctly.

**num_trials:**

This defines the number of neural networks to train. The best neural network will be chosen out of these trials.
The higher this number is, the longer it will take to train, but there is a better chance of getting good
performance. The default is 10 trials.


The training script will produce 5 files and place them in runs/run_name:

**log.out:**

This file is the console log from the training script. It shows the progression of training, the neural net
architecture used, and the final accuracy results from the run.

**model.pt:**

This file is the trained neural network. During inference, this file will be loaded in order to predict the class of
new images.

**class_order.csv:**

This is a very important file. Without this file the class predictions would be unusable. In short, the neural
network predicts class as an index. class_order.csv provides the class-index mapping specific to this
run's neural network. Nothing special needs to be done, just make sure that this file isn't moved from the folder.

**config.txt:**

This file gives the following information about the trained model:
- name
- date/time created
- accuracy
- field codes it was trained on

**best_predictions.csv:**

At the time of writing this software package, the amount of labeled images available was small. That means that
the amount of images available to hold out for testing the system was also small. This file shows the predictions
of the saved model on the entirety of the held out test images.

# Testing

When it's time to predict classes of new images, we need to run inference.py. This script also has 3
optional arguments.

**run_name:**

Name of the inference run and folder where the predictions are stored.

**dataset_directory:**

This is the path to the inference images. There isn't anything special that needs to be formatted about this
folder. It should just be a collection of images.

**model_directory:**

This argument tells the script which neural network to load. The name should match a previous training run
name. It will look in runs/model_directory for what it needs. Changing the directory that it looks for the model
should be easy if needed. Look on line 22 (under "global parameters").

Once inference.py is done, 2 files will be created and place them in runs/run_name:

**log.out:**

Similar to the log.out from the training script, but much less informative. It will print some errors if something
goes wrong.

**predictions.csv:**

This file is "gold" from the neural net. It's the whole reason why this package was written. The format is simple.
Each row of the file corresponds to an inference image and the associated class labels predicted by the model:

    img1.jpg,tower,fence,road
    img2.jpg,road
    img3.jpg,road,gravel
    img4.jpg,stake
    img5.jpb,cab,gravel,fence,tower
    ...
    ...

For the single-label version, there will only be one class prediction per image.

# Notes

**NOTE ON OPTIMAL HARDWARE:**

In order to run these scripts in a reasonable amount of time, a graphics processing unit (GPU) is essential. The main
computation when rendering graphics is the same as with running neural networks: matrix multiplications. For this
reason, it's standard to use as large of a GPU as you can afford to train and make predictions with neural networks.
Specifically, the standard is NVIDIA GPUs.

There are a variety of different options for computing with a GPU, each with different price and implementation
difficulty levels. The most important thing to remember is that you need an NVIDIA GPU with CUDA capability. CUDA
is just the middleware that ties neural network algorithms to GPU hardware. Luckily, all the recent GPU models put
out by NVIDIA will work for this. Here are some common GPU options you might see installed in a machine ready to
compute neural nets (from high-end to low-end):

* NVIDIA TITAN RTX [24GB of GDDR6 RAM]
* NVIDIA TITAN [12GB of RAM]
* NVIDIA TITAN Xp [12GB of GDDR5X RAM]
* NVIDIA GEFORCE RTX 2080 Ti [11GB of GDDR6 RAM]
* NVIDIA GEFORCE RTX 2080 SUPER [8GB of GDDR6 RAM]
* NVIDIA GEFORCE RTX 2080 [8GB of GDDR6 RAM]

Generally, the larger the size of RAM and the higher the GDDR_ number, the faster the GPU will work for training the
neural network.

The other option besides purchasing a machine equipped with a GPU is running the scipts on a cloud GPU. Every
major cloud compute platform has this capability front and center, which makes it a quick solution. The downside is
that you don't own the hardware and you have to pay according to your usage. As the training set enlarges,
computation time, storage requirements, and cost to compute enlarges as well. To get an idea of the computation
time required, with about 10 labeled examples per field code and 8 field codes, it takes my desktop computer about
30-45 minutes to train the neural network. My desktop computer is equipped with 11GB of GDDR5X RAM. Paying
for cloud compute resources could lower the training time, but in my experience with the free demos of cloud
compute GPUs there wasn't much of a difference in training time.


**MISC THOUGHTS:**

-- One question to ask yourself while assigning labels to images is "How obvious is it that this label is in this
image?". It's important for the neural network to learn what makes a "fence" a "fence". If many of the images labeled
with "fence" just have a fence in a tiny portion of the background, this could hinder learning what a "fence" is. The
main idea is to think about "What are the main objects in this image?"







