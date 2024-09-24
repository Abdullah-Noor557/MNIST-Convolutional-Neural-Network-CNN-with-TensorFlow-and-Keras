This project demonstrates a simple convolutional neural network (CNN) built using TensorFlow and Keras to classify handwritten digits from the MNIST dataset. The model is trained to recognize digits (0-9) and is evaluated on test data. This project also includes breakpoints for debugging using the pdb module.

Table of Contents
Overview
Installation
Usage
Loading Data
Creating the Model
Compiling and Training
Evaluation
Model Saving
Contributing
License
Overview
This repository provides a simple implementation of a CNN that processes the MNIST dataset using three primary convolutional layers, max-pooling layers, and dense layers. Debugging breakpoints are included to allow step-by-step execution for debugging purposes.

Installation
To get started, you'll need to have Python installed (preferably version 3.7 or above) along with the required dependencies:

bash
Copy code
pip install tensorflow pdb
Usage
You can use this script to load the MNIST dataset, create a CNN model, train the model, and evaluate its performance. The script includes pdb.set_trace() breakpoints for debugging at various stages.

Loading Data
The load_data() function loads the MNIST dataset using TensorFlow's built-in datasets.mnist module. The dataset is preprocessed by normalizing pixel values to the range [0, 1].

python
Copy code
(train_images, train_labels), (test_images, test_labels) = load_data()
Creating the Model
The create_model() function creates a CNN model with the following architecture:

Three convolutional layers with ReLU activation
Max-pooling layers after the first and second convolution layers
A flattening layer to prepare data for the dense layers
Two fully connected layers: one with 64 units and ReLU activation, and the final output layer with 10 units and a softmax activation
python
Copy code
model = create_model()
Note: The model creation process includes a debugging breakpoint using pdb.set_trace().

Compiling and Training
The compile_and_train_model() function compiles the model using the Adam optimizer, sparse categorical crossentropy loss, and accuracy as the evaluation metric. It then trains the model for 5 epochs on the training dataset.

python
Copy code
model = compile_and_train_model(model, train_images, train_labels)
Note: There is a pdb.set_trace() breakpoint to debug the compilation and training process.

Evaluation
The evaluate_model() function evaluates the trained model on the test dataset and prints the test accuracy.

python
Copy code
evaluate_model(model, test_images, test_labels)
A pdb.set_trace() breakpoint is also set during evaluation for step-by-step debugging.

Model Saving
After training, the model is saved as an HDF5 file (mnist_cnn_model.h5) using the following command:

python
Copy code
model.save('mnist_cnn_model.h5')
Contributing
If you'd like to contribute to this project, feel free to submit pull requests or open issues with any improvements or bug fixes.

License
This project is licensed under the MIT License. See the LICENSE file for details.
