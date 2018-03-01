**Traffic Sign Recognition** 

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

[image1]: ./writeup/Dataset_Visualisation.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./writeup/Web_Image_Responses.png "Feature Maps"
[image4]: ./web-images/70.png "Traffic Sign 1"
[image5]: ./web-images/unlimited-speed.png "Traffic Sign 2"
[image6]: ./web-images/general-caution.png "Traffic Sign 3"
[image7]: ./web-images/ahead-only.png "Traffic Sign 4"
[image8]: ./web-images/priority-road.png "Traffic Sign 5"
[image11]: ./writeup/Chart1.png "Softmax Probabilities for Test Image1"
[image12]: ./writeup/Chart2.png "Softmax Probabilities for Test Image2"
[image13]: ./writeup/Chart3.png "Softmax Probabilities for Test Image3"
[image14]: ./writeup/Chart4.png "Softmax Probabilities for Test Image4"
[image15]: ./writeup/Chart5.png "Softmax Probabilities for Test Image5"

---
###Writeup / README

This project is essentially about identifying German traffic signs. Here is a link to my [project code](https://github.com/samunda/Udacity-Self-Driving-Car-Term1/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb).

###Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410 (12.7% of training examples)
* The size of test set is 12630 (36.3% of training examples)
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It shows an example training image per each class. It also indicates the number of training examples per each class.

![alt text][image1]

###Design and Test a Model Architecture

I analyzed the range of the image data values and found it to be in the domain [0.00 255.00]. I used the formula (X - 128.0) / 128.0 to normalize image data so that the domain is [-1.00 1.00] and mean is 0. This is because it improves numerical conditioning and the training.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6				    |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16				    |
| Fully connected		| output 120   									|
| RELU					|												|
| Dropout				|												|
| Fully connected		| output 84   									|
| RELU					|												|
| Dropout				|												|
| Fully connected		| output 43   									|
| Softmax				|           									|

The weights and biases of layers were randomly initialized using a truncated normal with a mu of 0 and sigma of 0.1.

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer with a learning rate of 0.001. The batch size was 128 and number of epochs was 10. Dropout was set to 50%.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.983
* validation set accuracy of 0.946
* test set accuracy of 0.922

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

We chose the well known LeNet architecture. It performed well with a high training accuracy and a reasonably high (meeting the expected level) validation accuracy without overfitting.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 70 km/h      		    | 0, Speed limit (20km/h) 									| 
| unlimited-speed     	| 6, End of speed limit (80km/h)							|
| general-caution		| 18, General caution								|
| ahead-only      		| 35, Ahead only					 				|
| priority-road			| 12, Priority road     							|

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 92.2%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 15th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a speed limit of 70 km/h (probability of 0.846), and the image does contain a 70 km/h speed limit sign. The top five soft max probabilities are shown in below bar chart. It shows the next likely (with a probability < 0.06) candidates are other speed signs.

![alt text][image11]

For the second image, the model predicts a end of 80 km/h speed limit sign (probability of 0.677). However, this is an incorrect classification as the sign actually contains an end of all speeds and passing limits sign. The model still predicts a reasonably high probability of 0.316 for the correct sign.

![alt text][image12]

For the third image, the model correctly predicts a general caution sign. Rest of the softmax probabilities show that other signs are relatively unlikely.

![alt text][image13]

For the fourth image, the model correctly predicts an ahead only sign. Rest of the softmax probabilities show that other signs are relatively unlikely.

![alt text][image14]

For the fifth image, the model correctly predicts a priority road sign. Rest of the softmax probabilities show that other signs are relatively unlikely.

![alt text][image15]



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

The figure below displays the output of the first layer (cnn1) of the model for the five test images. It shows that the first CNN layer has learnt to extract low level image features such as edges.

![alt text][image3]

