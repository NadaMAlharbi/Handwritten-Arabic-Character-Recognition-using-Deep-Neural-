# Handwritten Arabic Character Recognition using Deep Neural Networks

**Authors:** Nada Mohammed Alharbi  
**Project Type:** Academic Project for CCAI-435 - Deep Neural Networks  

## Project Overview
This project focuses on building a deep neural network model for recognizing handwritten Arabic characters. Due to the complexity of Arabic script and variations in handwriting, this is a challenging task in the fields of pattern recognition and computer vision. Accurate recognition of Arabic characters is essential for various applications, such as document processing, text digitization, and language translation. This project leverages Convolutional Neural Networks (CNNs) to achieve high accuracy in recognizing Arabic characters.

## Table of Contents
1. [Dataset](#dataset)
2. [Model Architecture](#model-architecture)
3. [Training and Evaluation](#training-and-evaluation)
4. [Results](#results)
5. [Conclusion](#conclusion)
6. [How to Run](#how-to-run)
7. [Dependencies](#dependencies)

## Dataset
The dataset used in this project is the **Arabic-Handwritten-Chars** dataset from Kaggle, which includes a total of **16,800 handwritten characters** contributed by 60 participants. Each participant provided multiple samples of each Arabic character (from 'alef' to 'yeh'), and the dataset is divided as follows:
- **Training Set:** 13,440 characters (480 images per character class)
- **Testing Set:** 3,360 characters (120 images per character class)

This dataset serves as a valuable resource for training and evaluating models specifically designed for Arabic character recognition.

## Model Architecture
The model architecture is based on Convolutional Neural Networks (CNNs), structured as follows:

1. **Convolutional Blocks:**
   - **First Block:** 64 filters with a 3x3 kernel, ReLU activation, batch normalization, max pooling (2x2), and dropout (0.3).
   - **Second Block:** 128 filters with a 3x3 kernel, ReLU activation, batch normalization, max pooling (2x2), and dropout (0.3).
   - **Third Block:** 256 filters with a 3x3 kernel, ReLU activation, batch normalization, max pooling (2x2), and dropout (0.4).

2. **Classifier Head:**
   - **Flatten Layer:** Converts 2D feature maps to a 1D vector.
   - **Dense Layer:** 512 units with ReLU activation and dropout (0.5).
   - **Output Layer:** 28 units with softmax activation, corresponding to the 28 Arabic character classes.

3. **Total Layers:** 16 layers, including convolutional blocks, max-pooling layers, dropout layers, and the classifier head.

## Training and Evaluation
The model was trained using the following parameters:
- **Epochs:** Up to 100, with early stopping based on validation loss to avoid overfitting.
- **Batch Sizes:** 32 and 64 were tested, with batch size 32 yielding better results.
- **Callbacks:** Early stopping was implemented to restore the model to its best weights when the validation loss stopped improving.

### Performance Comparison:
- **Batch Size 32:** Achieved higher training accuracy (93.35%) and validation accuracy (92.89%), with lower training loss, providing better generalization.
- **Batch Size 64:** Showed slightly lower accuracy and higher training loss compared to batch size 32.

## Results
- **Test Accuracy:** The model achieved a high test accuracy of approximately **91.85%** with batch size 32, demonstrating effective learning without overfitting.
- **Training and Validation Curves:** The training and validation accuracy curves showed that the model was able to learn effectively from the data.
- **Optimal Performance:** Batch size 32 provided better results, indicating improved generalization on unseen data.

## Conclusion
The project successfully demonstrates the capability of CNNs in recognizing handwritten Arabic characters with a high degree of accuracy. By utilizing a robust CNN architecture and effective hyperparameter tuning, the model achieved excellent performance, paving the way for future applications in Arabic language processing, document digitization, and multilingual translation.

## How to Run
1. **Environment Setup:** Ensure you have Python and the necessary libraries (`TensorFlow`, `Keras`, `NumPy`, `Matplotlib`) installed.
2. **Dataset Preparation:** Download the "Arabic-Handwritten-Chars" dataset from Kaggle and place it in your working directory.
3. **Execute the Code:** Run the notebook `Handwritten Arabic Character Recognition using Deep Neural Networks.ipynb` to train and evaluate the model.

## Dependencies
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
