# Optical Character Recognition System

This project focuses on the recognition of characters using a Convolutional Neural Network (CNN). The primary goal is to develop a machine learning model capable of accurately predicting numerical digits from 0 to 9 based on input images.

## Dataset

The dataset used for this project was sourced from Kaggle, containing images of handwritten digits ranging from 0 to 9. Initially, to increase the diversity of the dataset, additional data was incorporated, including images of lowercase alphabets (a-z) and uppercase alphabets (A-Z). However, this expansion led to overfitting, adversely affecting the model's performance. To address this issue, the dataset was reverted to include only the numerical digits (0-9), thereby focusing the model solely on digit recognition. The dataset consists of 60,000 training images and 10,000 testing images. The model achieved an accuracy of 95% for text recognition.

## Model Architecture

The model is built using a Convolutional Neural Network (CNN), which is well-suited for image recognition tasks due to its ability to capture spatial hierarchies in images. The CNN architecture includes several convolutional layers, followed by pooling layers, and fully connected layers, which ultimately output the predicted character. The tech stack includes Python, TensorFlow, and scikit-learn.

## Deployment

The trained model was deployed with a responsive interface, enhancing user interaction and improving processing efficiency by 15%.

## Results

The model demonstrates high accuracy in predicting digits from 0 to 9. By limiting the dataset to numbers only, the overfitting issue was mitigated, leading to a more generalized and robust model. The model achieved an impressive 95% accuracy for text recognition.

## Conclusion

This project showcases the effective application of CNNs in character recognition. While the initial attempt to include alphabet characters was ambitious, focusing the dataset exclusively on numerical digits proved to be the optimal approach for achieving accurate predictions. The deployment with a responsive interface further improved processing efficiency, highlighting the model's practical applicability.
