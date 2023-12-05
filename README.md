# MediCURE Disease Prediction based on Symptoms

### About the project:
The accurate analysis of medical database benefits in early disease prediction, patient care, and community services is critical. This project proposes a web application for the users to identify the disease they have been infected with based on the symptoms they have been suffering with a given confidence score. It also includes the capability of recommending home remedies as a treatment for the disease that the user may be infected with.  


### Working of project:

https://user-images.githubusercontent.com/107244393/235594013-79ac30cd-e462-4601-85eb-774a14658ab7.mp4

### Objectives:
•	To create machine learning models that will be used to predict disease/disease the user may be suffering from based on the symptoms we have used in our dataset.

•	Create a Web-Application for the model using Flask.

•	Features some additional features that will recommend home remedies as a treatment for the predicted diseases. 

•	Deploy the application on a cloud platform, AWS.

•	The application is secured using AWS WAF and AWS Application Load Balancer.

•	Deploy the application on a Docker container to streamline the development lifecycle.

•	Comparative Analysis of models like Logistic regression, SVM, Multilayer Perceptron, ANN, Naive-Bayes, Decision-tree, and some variations of decision tree like XGBoost, etc., to check for the best suitability for predicting the disease.

•	This will help people get a better idea of what they may be suffering from instead of a Random Google search that predicts even symptoms of the common cold as something grave but we need to remember that this tool does not provide medical advice. It is intended for information purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment

### Methodology:
![image](https://user-images.githubusercontent.com/107244393/235595779-6f6355eb-1b5f-4193-aa82-6d0ee5e6fb53.png)

### Dataset
The dataset is taken from Kaggle as secondary data. It comprises of 8836 rows and 489 columns. The dataset consists of 261 distinct diseases along with their symptoms. Each disease is treated as the label and all symptoms are treated as specific attributes or columns. To multiply the dataset, each disease’s symptoms are picked up, combinations of the symptoms are created and added as new rows in the dataset.
The dataset used are in the Dataset Folder.
The dataset for Treatment is named as cure_minor.xlsx

### Deployment
During our development phase, we used dev containers which will deploy the project built on so that it provides a reliable and consistent environment throughout the project. The complete application is also deployed on docker, an image is built and pushed to Docker Hub from where any user can pull the image, helped to remove all the dependency issues that might arise due to differences in the environment. We have also created an EC2 instance and then a target group for the created  EC2 instance. Configuring the request protocol and setting it to HTTP/1.1 helps to easily and reliably communicate between WAF and EC2. After the integration of WAF, we will be using Application Load Balancers so as to easily distribute the workload. We then associated our web ACL (Access Control List) with the load balancers to further add an extra layer of security in our application.
![image](https://user-images.githubusercontent.com/107244393/235596568-b828950e-68c0-4499-817c-d7a2faca64b5.png)
![image](https://user-images.githubusercontent.com/107244393/235596585-9c501ce2-f2da-4431-bfe4-49f21b93e2c4.png)
![image](https://user-images.githubusercontent.com/107244393/235596629-7a18d4f5-14b9-4525-8b08-582372b0d40d.png)








