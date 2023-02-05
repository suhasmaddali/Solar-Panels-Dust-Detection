# 🌞 Solar Panels Dust Detection

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org)  [![](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org) [![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/) [![](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white)](https://www.scipy.org) [![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) [![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)  [![](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com) [![](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io) [![](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com)

## Introduction 

__Solar panels__ are used in quite a large number of industries. Examples include residential, agricultural, manufacturing, healthcare and retail industries. As these panels are used for many years, there is often dust accumulated on their surface due to factors such as climate, trees, vegetation, lack of maintenance and bird droppings. Since the primary source of energy for solar cells comes from the sun, the dust accumulated on the surface of the panels lead to reduced efficiency, reduced lifespan, increased costs and inefficient resource use. 

<img src = "https://github.com/suhasmaddali/Images/blob/main/Solar%20Panel%20GitHub%20Images.jpg" />

## Challenges

There are often challenges associated with using solar panels for generation of electricity. One of the primary issues is their low efficiency. However, they are quite useful in areas where there is abundant sunlight. Due to dust accumulated on their surface, their efficiency reduces as it is not able to absorb the solar radiation from the sun. 

## Deep Learning and Data Sciencie

With the rise and development of deep learning models used for recognizing images, it is possible to find a solution to this problem of accumulation of dust on solar panels. Convolutional Neural Networks (CNNs) are used for image processing and can extract useful insights from them and make prediction about whether solar panels are dusty or clean. In the recent decade, there has been a rapid growth of transformers and how they could be used for vision tasks (known as vision transformers). There are other learning strategies such as transfer learning that also result in improved performance and minimal training. Therefore, it is possible to improve the systems that detect dust on the surface of solar panels, leading to improved efficiency and reduced operational and maintenance costs. 

## Exploratory Data Analysis (EDA)

Performing exploratory data analysis could reveal interesting patterns about the data (images) so that adequate feature engineering could be performed, leading to improved model performance. After exploring the images, it could be seen that there are a lot of diverse and inconsistent images of dusty solar panel images. Therefore, collecting more data that is consistent and indicative of dusty panels is an important step. Since the data that we are dealing with is quite small, there is a higher chance of overfitting, especially if using complex models through transfer learning. All of these must be followed to ensure that there is increased value by using our product to detect the dusty solar panels. 

## Feature Engineering

## Metrics

As there are 2 classes (clean or dusty), it becomes a classification problem. We would be using binary cross entropy loss and modify the weights of deep neural networks until the desired performance is met. Below are the metrics used to track various model performances. 

1. Accuracy
2. Precision
3. Recall
4. F1-Score
5. ROC-AUC Curves

## Deep Learning Models

## 💻 Training with NVIDIA's RTX 2080 graphics card for Computer Vision Tasks 

* __GPU-accelerated deep learning frameworks__ offer flexibility to design and train __deep neural networks__.
* With __cuDNN__ and __Nvidia's graphics drivers__, I was able to train the models really quickly by using __GPU__ cores rather than the __CPU__ cores.
* This led to a significant increase in the __speed of training__ and __developing convolutional neural networks (CNNs)__. 

## 👉 Directions to download the repository and run the notebook 

This is for the Washington Bike Demand Prediction repository. But the same steps could be followed for this repository. 

1. You'll have to download and install Git which could be used for cloning the repositories that are present. The link to download Git is https://git-scm.com/downloads.
 
&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(14).png" width = "600"/>
 
2. Once "Git" is downloaded and installed, you'll have to right-click on the location where you would like to download this repository. I would like to store it in the "Git Folder" location. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(15).png" width = "600" />

3. If you have successfully installed Git, you'll get an option called "Gitbash Here" when you right-click on a particular location. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(16).png" width = "600" />


4. Once the Gitbash terminal opens, you'll need to write "Git clone" and then paste the link to the repository.
 
&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(18).png" width = "600" />

5. The link to the repository can be found when you click on "Code" (Green button) and then, there would be an HTML link just below. Therefore, the command to download a particular repository should be "Git clone HTML" where the HTML is replaced by the link to this repository. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(17).png" width = "600" />

6. After successfully downloading the repository, there should be a folder with the name of the repository as can be seen below.

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(19).png" width = "600" />

7. Once the repository is downloaded, go to the start button and search for "Anaconda Prompt" if you have anaconda installed. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(20).png" width = "600" />

8. Later, open the Jupyter notebook by writing "Jupyter notebook" in the Anaconda prompt. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(21).png" width = "600" />

9. Now the following would open with a list of directories. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(22).png" width = "600" />

10. Search for the location where you have downloaded the repository. Be sure to open that folder. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(12).png" width = "600" />

11. You might now run the .ipynb files present in the repository to open the notebook and the python code present in it. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(13).png" width = "600" />

That's it, you should be able to read the code now. Thanks. 
