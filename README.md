# ecotup-ML
This repository for ecotup project focusing on Machine Learning and Python script, in this repository we have 3 model which is the trash classification softmax, trash classification sigmoid, and we also have a clustering model that fused with sorting algoritm especially using TSP Greedy Algorithm. For the Trash Classification Model we using the InceptionV3 transfer learning using Imagenet weight for more accurate prediction.

## Introduction
Machine learning is a computer systems that are able to learn and adapt without following explicit instructions, by using algorithms and statistical models to analyze and draw inferences from patterns in data. With this technology almost any problem that humans have a hard time doing can be solved easily and it can make humans work much more easier. Our job as a machine learning developer in this Ecotup team is to make and give innovation that can make the app much more automatic, more efficient, can help user become better and much more satisfying to use.  

## Datasets that we use
- [Softmax trash dataset make by Rivaldo uploaded to kaggle](https://www.kaggle.com/datasets/rivaldo1233/reallife-trash-dataset)
- [Sigmoid trash dataset make by Sashaank Sekar uploaded to kaggle](https://www.kaggle.com/datasets/techsash/waste-classification-data)
  
## Model that we use 
- [Sigmoid trash classification model code](https://github.com/ecotup/ecotup-ML/blob/main/Model_Code/Trash_Classification_Sigmoid.ipynb)
- [Softmax trash classification model code](https://github.com/ecotup/ecotup-ML/blob/main/Model_Code/Trash_Classification_Softmax.ipynb)
- [Clustering user houses model code](https://github.com/ecotup/ecotup-ML/blob/main/Model_Code/Clustering_houses_AI.ipynb)

## About the Model 
The model that we use are Softmax and Sigmoid Trash Classification about the sigmoid trash classification were actually using a softmax output but because its only 2 results were saying sigmoid trash classification not softmax. the trash classification model is using transfer learning for better efficiency and better prediction, we're using the InceptionV3 with Imagenet weights and added couple of more layers for training for achieving the best result. We're also using Clustering model that can group people houses before being sorted later on the reason were not using the model file and were using more like a script code is because our data is updating real time making it really impossible to cluster the users properly for the subscription feature.

## Script that we use 
- [Find nearest driver script](https://github.com/ecotup/ecotup-ML/blob/main/Feature_python_dockerize/finding_driver.py)
- [Clusterina and sorting the user houses script](https://github.com/ecotup/ecotup-ML/blob/main/Feature_python_dockerize/clustering_and_sort.py)
- [Script for combining both of the script routes](https://github.com/ecotup/ecotup-ML/blob/main/Feature_python_dockerize/flask_app_capstone.py)

## About the Script
One of the Script here is for finding the nearest driver for the user in the one time only pick up. The first thing it do is to take the user id that already been sent to http parameter when running then the algorithm is using the SQLAlchemy to use the SQL engine for accessing the SQL that our Cloud Computing is working on and the pymysql is for running the query the query is used to take the user data longitude and latitude and then transfer it to another query that take the necessary data from driver id for calculation using haversine calculation and getting the nearest driver in the SQL database. Then it will return the necessary data to the android. The second one contain the model for clustering and sending the data and cluster it up then sort it using the TSP Greedy Algorithm for sorting it based on the nearest point and then sorting it in order. We also have the last script called flask-app-capstone for combining both of the routes and can be deployed on the same port using blueprint syntax that already exist in the Flask library.

## Folder list inside the repository and whats inside
- feature_python_dockerize = inside is fill with features that already been prepared using dockerfile that only need to build image then deploy using google cloud run
- Model_Code = inside is fill with the model code that already succeeded and already turned into the model file while also implemented to the app

## Evaluation 
All and all we confident that our work have succeeded and getting the most amazing result possible, with the Softmax and Sigmoid Trash Classification that reached more than 90% in training and at least more than 80% in validation. For the cluster model were finding it satisfactory and yieding impressive result! The script also help a lot at the Cloud Computing and Mobile Development side. 

## Further Work
We're already done working on the voice recognition model. But because of Mobile Development huge workload we decided to implement and deploy it later on after we're done with this capstone. We're also planning to make our own routing for the one time pick up and subscription without relying on Google Routing because of the budget. Then we're planning on making Script/AI for recommendation about article related to trash and nature itself. This team also want to developed a AI for calculating how much fertilizer and water needed for plants and disease image classification on plants. 

## Afterwords
We're now at the end of the documentation! We tackle of many aspect of our work and our ambition moving forward! we thank you for your dedication for reading this documentation about our ML repository, and lastly goodbye :D.
