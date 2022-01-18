# Single-Image-Object-Counting-and-Localizing-using-Active-Learning

This is the implementation code of the paper: **[Single-Image-Object-Counting-and-Localizing-using-Active-Learning](https://www.cs.huji.ac.il/w~inbarhub/projects/count_WACV/paper.pdf)** presented in WACV 2022.

The dataset used in this paper is available at [Single Image Counting Dataset](https://github.com/inbarhub/single_image_dataset).

### Abstract
The need to count and localize repeating objects in an image arises in different scenarios, such as biological microscopy studies, production-lines inspection, and surveillance recordings analysis. The use of supervised Convolutional Neural Networks (CNNs) achieves accurate object detection when trained over large class-specific datasets. The labeling effort in this approach does not pay-off when the counting is required over few images of a unique object class.

We present a new method for counting and localizing repeating objects in single-image scenarios, assuming no pretrained classifier is available. Our method trains a CNN ver a small set of labels carefully collected from the input mage in few active-learning iterations. At each iteration, he latent space of the network is analyzed to extract a minimal number of user-queries that strives to both sample the n-class manifold as thoroughly as possible as well as avoid edundant labels.
Compared with existing user-assisted counting methods, our active-learning iterations achieve state-of-the-art performance in terms of counting and localizing accuracy, number of user mouse clicks, and running-time. This evaluation was performed through a large user study over a wide range of image classes with diverse conditions of illumination and occlusions

### Running the code

You can run the code by calling to count_repetitive_objects.py script with two arguments: an image name and the participant name.
For example:
* python3 count_repetitive_objects.py CellSml Inbar
* python3 count_repetitive_objects.py Logs Jonathan
* python3 count_repetitive_objects.py Water Michael

The names of the images (e.g., CellSml, Logs...) can be found in the dataset or in Table 1 in the paper.

At every iteration, positive and negative potential locations will be shown for user feedback. The user needs to click on the *misclassified* ones and close the window. The color of the windows that are clicked by the user will be changed to black.

This prodeucde is repeated 5 times. At the end, the final repeaintg locations will be presented to the user. The localization errors and the counting will be shown on the console/termial as well as will be saved in a dedicated folder.

To simplify, we removed the option of marking the initial repeating object. You can see the windows locations in conf.py script.

More information can be found in the paper. If you have any questions, you can find me in: inbarhub@gmail.com
