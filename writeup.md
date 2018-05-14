# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report_images/1.png "Visualization"
[image2]: ./report_images/2.JPG "speckle noise"
[image3]: ./report_images/3.JPG "Augemented data"
[image4]: ./download_images/1.jpg "Traffic Sign 1"
[image5]: ./download_images/2.jpg "Traffic Sign 2"
[image6]: ./download_images/3.jpg "Traffic Sign 3"
[image7]: ./download_images/4.jpg "Traffic Sign 4"
[image8]: ./download_images/5.jpg "Traffic Sign 5"
[image9]: ./report_images/9.JPG "Individual performance"
[image10]: ./report_images/10.JPG "Feature Map"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

Here is a link to my [project code](https://github.com/Awesomex005/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. It is a bar chart showing how the data set distributed.

![alt text][image1]

### Design and Test a Model Architecture

I normalized the image data to because big input data could produce big output logits, which could lead to opinionated initial prediction, the model could be hard to optimize.

I decided to generate additional data because some type of signs gets relatively small quantity of samples, the model may overfitting on such type of signs.

To add more data to the the data set, I used techniques like adding slitghtly noise and translatin.

Here is an example of an original image and an augmented image:

![alt text][image2]

The visualization of augmented data set.
![alt text][image3]

#### Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 				    |
| Flatten               | outputs 800                                   |
| Fully connected		| outputs 240  									|
| RELU					|												|
| Dropout				| keep_prob	0.8 								|
| Fully connected		| outputs 168  									|
| RELU					|												|
| Dropout				| keep_prob	0.8 								|
| Fully connected		| outputs 43  									|
| Softmax				|        						    			|
 
To train the model, I used an AdamOptimizer, 128 batch size, 20 epochs and learning rate at 0.001

My final model results were:
* training set accuracy of 0.862
* validation set accuracy of 0.965   
* test set accuracy of 0.9414885195398368

Here is a visualization of test set accuracy of individual sign type.
![alt text][image9]

The initial model
* The first architecture I used is LeNet5 because it is simple, easy to use and understand.
Problem
* LeNet5 doing so well in fitting train set and finally overfitting on the train set.
Solution
* I add dropout layer in the last 2 full connection hidden layers. 
Dropout randomly drop neurons, I think this is similar to trainning different models when every new batch fed in, in the end we get a combination model from many different models. 
In this way, the model is forced to learn from different ‘aspects’ of the data and learn from different mistakes. This helps to produce an stronger classifier, which is less prone to overfitting. I think this is a kind of *ensemnbe learning*. 

### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)      		| Speed limit (30km/h)   									| 
| Yield     			| Yield 										|
| Beware of ice/snow					| Beware of ice/snow											|
| Priority road	      		| Priority road					 				|
| Road work		| Road work      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.1%. According to image performance of sign types, if I fed in Double curve or Pedestrians signs, the performance could be lower.

The code for making predictions on my final model is located in the 21th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Speed limit (30km/h) (probability of 0.99). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00        			| Speed limit (30km/h)   									| 
| 3.05116510e-09     				| Speed limit (50km/h) 										|
| 5.21126940e-19					| Speed limit (60km/h)											|
| 1.64775428e-21	      			| Speed limit (20km/h)					 				|
| 6.13898473e-22				    | Speed limit (80km/h)     							|

#### Visualization of network's feature maps
Here is a visualization of conv1 feature maps with
![alt text][image10]
I don't have much intuition on this feature maps.

