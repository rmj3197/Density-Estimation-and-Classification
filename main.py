#!/usr/bin/env python
# coding: utf-8

# In[49]:


import numpy as np
import scipy.io
import scipy.stats as stats
from skimage.io import imshow
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Importing the Dataset for this project

# In[50]:


dataset = scipy.io.loadmat("C:/Users\Raktim\Desktop\ASU\SML\mnist_data.mat")
train_dataset=dataset["trX"]
test_dataset=dataset["tsX"]
label_train_dataset=dataset["trY"]
label_test_dataset=dataset["tsY"]


# In[51]:


#splitting the train dataset into Class - 7 and Class - 8
train_dataset_7=train_dataset[0:6265]
train_dataset_8=train_dataset[6265:]


# Mentioned in the description that 6265 images in the dataset are 7 and 5851 images in the dataset are 8. Just plotting couple of images for confirmation

# In[52]:


imgplot = plt.imshow(np.reshape(train_dataset[6264],(28,28)),cmap=plt.cm.gray)


# In[53]:


imgplot = plt.imshow(np.reshape(train_dataset[12115],(28,28)),cmap=plt.cm.gray)


# The images given in the dataset have 784 features (28 x 28). Since in this project it has been mentioned to work with mean and standard deviation as features, we are going to calculate mean and standard deviation of pixel values for each image for both the Training and Testing dataset

# Mean and SD arrays for Train Dataset

# In[54]:


train_mean_images=np.mean(train_dataset,axis=1)
train_sd_images=np.std(train_dataset,axis=1)


# Mean and SD arrays for Testing Dataset

# In[55]:


test_mean_images=np.mean(test_dataset,axis=1)
test_sd_images=np.std(test_dataset,axis=1)


# Calculating the Mean and Standard Deviation of each image belonging to class - 7 and class - 8

# In[56]:


mean_images_7=np.mean(train_dataset_7,axis=1)
sd_images_7=np.std(train_dataset_7,axis=1)
mean_images_8=np.mean(train_dataset_8,axis=1)
sd_images_8=np.std(train_dataset_8,axis=1)


# Assuming the features - Mean and Standard Deviation - follow a Gaussian Distribution, we plot the distribution for the complete training dataset and subsequently for Class - 7 and Class - 8 images.

# In[57]:


#plotting Gaussian Distribution for Complete Training Dataset for Feature - Mean
pdf_mean_images=stats.norm.pdf(np.sort(train_mean_images),train_mean_images.mean(),train_mean_images.std())
plt.plot(np.sort(train_mean_images),pdf_mean_images)
plt.hist(np.sort(train_mean_images),density=True)


# In[58]:


#plotting Gaussian Distribution for Complete Training Dataset for Feature - Standard Deviation
pdf_sd_images=stats.norm.pdf(np.sort(train_sd_images),train_sd_images.mean(),train_sd_images.std())
plt.plot(np.sort(train_sd_images),pdf_sd_images)
plt.hist(np.sort(train_sd_images),density=True)


# In[59]:


#plotting Gaussian Distribution for Images of class - 7 for feature - Mean
pdf_mean_images_7=stats.norm.pdf(np.sort(mean_images_7),mean_images_7.mean(),mean_images_7.std())
plt.plot(np.sort(mean_images_7),pdf_mean_images_7)
plt.hist(np.sort(mean_images_7),density=True)


# In[60]:



#plotting Gaussian Distribution for Images of class - 7 for feature - Standard Deviation
pdf_sd_images_7=stats.norm.pdf(np.sort(sd_images_7),sd_images_7.mean(),sd_images_7.std())
plt.plot(np.sort(sd_images_7),pdf_sd_images_7)
plt.hist(np.sort(sd_images_7),density=True)


# In[61]:


#plotting Gaussian Distribution for Images of class - 8 for feature - Mean
pdf_mean_images_8=stats.norm.pdf(np.sort(mean_images_8),mean_images_8.mean(),mean_images_8.std())
plt.plot(np.sort(mean_images_8),pdf_mean_images_8)
plt.hist(np.sort(mean_images_8),density=True)


# In[62]:


#plotting Gaussian Distribution for Images of class - 8 for feature - Standard Deviation
pdf_sd_images_8=stats.norm.pdf(np.sort(sd_images_8),sd_images_8.mean(),sd_images_8.std())
plt.plot(np.sort(sd_images_8),pdf_sd_images_8)
plt.hist(np.sort(sd_images_8),density=True)


# NAIVE BAYES CLASSIFIER

# We consider that each feature is conditionally independent of every other feature.

# In[63]:


#calculating the probability that an image belongs to class 7 and class 8 respectively
p_class_7=mean_images_7.size/train_mean_images.size
p_class_8=mean_images_8.size/train_mean_images.size


# Declaring the Univariate Normal Distribution (PDF) [P(y/x)] where y represents the label and x represent the features

# In[64]:


def p_x_given_y(x,mean, variance):
    p = (1/(np.sqrt(2*np.pi*variance)))*np.exp(-(x-mean)**2/(2*variance))
    return p


# Calculating the numerator for posterior probability for class - 7 

# In[65]:


numr_postr_class_7 = p_class_7*p_x_given_y(test_mean_images,mean_images_7.mean(),mean_images_7.var())*p_x_given_y(test_sd_images,sd_images_7.mean(),sd_images_7.var())


# Calculating the numerator for posterior probability for class - 8

# In[66]:


numr_postr_class_8 = p_class_8*p_x_given_y(test_mean_images,mean_images_8.mean(),mean_images_8.var())*p_x_given_y(test_sd_images,sd_images_8.mean(),sd_images_8.var())


# If (numr_postr_class_7) > (numr_postr_class_8), then the image is classified as belonging to Class - 7. 
# 
# If (numr_postr_class_8) > (numr_postr_class_7), then the image is classified as belonging to Class - 8. 
# 

# In[67]:


comparison_numr=np.greater(numr_postr_class_8,numr_postr_class_7)
comparison_numr_int=comparison_numr.astype(np.int)


# The comparision_numr is a array of True and False values. Converting this into int type will give False = 0 and True = 1.
# 
# If (numr_postr_class_8) > (numr_postr_class_7) is False, then the image belongs to class 7.  Hence the predicted label for this image is "0" which represents that this image belongs to Class - 7. 
# 
# If (numr_postr_class_8) > (numr_postr_class_7) is True, then the image belongs to class 8.  Hence the predicted label for this image is "1" which represents that this image belongs to Class - 8. 
# 
# We can compare the comparison_numr_int array with the label_test_dataset array to findout how many images in the test dataset have been correctly predicted
# 

# In[68]:


comparison_between_int_and_label=np.equal(comparison_numr_int,label_test_dataset)


# Now counting the number of True values

# In[69]:


count_correct_prediction = np.count_nonzero(comparison_between_int_and_label)


# The overall accuracy can be defined as the (Total Number of Images Predicted Correctly of the Test Dataset/Total Number of Images in the Test Dataset)

# In[70]:


accuracy = (count_correct_prediction/label_test_dataset.size)*100
print( "The Accuracy of the Naive Bayes Classifier is ", accuracy,"%")


# We can also calculate the prediction accuracy for Class - 7 and Class - 8 separately

# In[71]:


#calculating the accuracy for class - 7
accuracy_7=(np.count_nonzero(np.equal(comparison_numr_int[0:1028],np.squeeze(label_test_dataset)[0:1028]))/np.squeeze(label_test_dataset)[0:1028].size)*100
print( "The Accuracy of the Naive Bayes Classifier for predicting class -7 images only is ", accuracy_7,"%")


# In[72]:


#calculating the accuracy for class - 8
accuracy_8=(np.count_nonzero(np.equal(comparison_numr_int[1028:],np.squeeze(label_test_dataset)[1028:]))/np.squeeze(label_test_dataset)[1028:].size)*100
print( "The Accuracy of the Naive Bayes Classifier for predicting class -8 images only is ", accuracy_8,"%")


# LOGISTIC REGRESSION CLASSIFIER

# In[73]:


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# In[74]:


def log_likelihood(X,y,theta):
    z = np.dot(X,theta)
    log_likelihood_val = np.sum( y*z - np.log(1 + np.exp(z)) )
    return log_likelihood_val


# In[75]:


def logistic_regression(X, y, num_iterations, learning_rate, add_intercept = False):
    
    if add_intercept:
        intercept = np.ones((X.shape[0], 1))
        X = np.hstack((intercept, X))
        
    theta = np.zeros(X.shape[1])
    
    for step in range (num_iterations):
        z = np.dot(X, theta)
        preds = sigmoid(z)

        diff = y - preds
        
        gradient = np.dot(X.T, diff)
        theta += learning_rate * gradient

        if step % 10000 == 0:
            print (log_likelihood(X, y,theta))
        
    return theta


# In[76]:


train_images_features = np.column_stack((train_mean_images,train_sd_images))
test_images_features = np.column_stack((test_mean_images,test_sd_images))
train_images_labels=np.squeeze(label_train_dataset.transpose())
test_images_labels=np.squeeze(label_test_dataset.transpose())


# In[77]:


thetas = logistic_regression(train_images_features, train_images_labels,num_iterations = 100000, learning_rate = 0.001, add_intercept=True)
print(thetas)


# Prediction

# In[78]:


final_z = np.dot(np.hstack((np.ones((test_images_features.shape[0], 1)),
                                 test_images_features)), thetas)
preds = np.round(sigmoid(final_z))


# In[79]:


accuracy_lr= (preds == test_images_labels).sum().astype(int)/len(test_images_labels)


# In[80]:


print ('The accuracy of Logistic Regression Classifier is ',accuracy_lr*100, '%')


# In[81]:


accuracy_lr_7= (preds[0:1028] == test_images_labels[0:1028]).sum().astype(int)/len(test_images_labels[0:1028])
print ('The accuracy of Logistic Regression Classifier for Class - 7 is ',accuracy_lr_7*100, '%')


# In[82]:


accuracy_lr_8= (preds[1028:] == test_images_labels[1028:]).sum().astype(int)/len(test_images_labels[1028:])
print ('The accuracy of Logistic Regression Classifier for Class - 8 is ',accuracy_lr_8*100, '%')


# In[ ]:




