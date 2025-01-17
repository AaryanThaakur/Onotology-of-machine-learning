import pyirk as p

__URI__ = "irk:/oml_work/1.0"
keymanager = p.KeyManager()
p.register_mod(__URI__, keymanager)
p.start_mod(__URI__)


#  a Perceptron is composed of two parts: “linear part” and “activation function”
#  the output of the linear part is the input of the activation function
#  ReLU is often used as activation function

I1015 = p.create_item(
    "I1015",
    R1__has_label="Mathematical Function",
    R2__has_description="A mathematical function is a relation between a set of inputs and a set of possible outputs. It takes an input (or multiple inputs) and produces an output based on a specific rule or computation. In machine learning, mathematical functions are used to model relationships between features and target variables, and they are central to algorithms, including those used in neural networks."
)
I1026 = p.create_item(
    "I1026",
    R1__has_label="Perceptron",
    R2__has_description="A perceptron is a type of artificial neural network unit or node that performs a simple linear transformation followed by an activation function."
      "It takes multiple inputs, applies weights to them, sums the weighted inputs, and passes the result through an activation function to produce an output. Perceptrons are the building blocks of more complex neural networks."
)

I1027 = p.create_item(
    "I1027",
    R1__has_label="Linear Part",
    R2__has_description="The linear part of a neural network refers to the linear combination of inputs, weights, and biases that are computed before passing through an activation function."
      "This part involves calculating the weighted sum of the inputs, which is then used to determine the output of the neuron or perceptron.",
    R5__is_part_of=I1026["Perceptron"]
    R15__is_element_of=I1015["Mathematical Function"]
)

I1028 = p.create_item(
    "I1028",
    R1__has_label="Activation Function",
    R2__has_description="An activation function is a mathematical function applied to the output of a neuron to introduce non-linearity into the model." 
    "It helps the model learn complex patterns and behaviors by transforming the input from the linear combination of features. Common activation functions include Sigmoid, Tanh, and ReLU.",
    R5__is_part_of=I1026["Perceptron"]

)

I1029 = p.create_item(
    "I1029",
    R1__has_label="ReLU",
    R2__has_description="ReLU (Rectified Linear Unit) is an activation function commonly used in deep learning models."
      "It outputs the input directly if it is positive; otherwise, it outputs zero. ReLU introduces non-linearity into the model, allowing it to learn more complex patterns, and is computationally efficient, making it widely used in neural networks.",
    R15__is_element_of=I1015["Mathematical Function"]
)


# 'Convolutional Neural Networks' process 'images' to 'extract features'.

I1004 = p.create_item(
    "I1004",
    R1__has_label="Deep Learning Model",
    R2__has_description="A deep learning model is a type of neural network with multiple layers between the input and output layers."
        " These models are designed to automatically learn complex patterns and representations in data, especially for tasks such as image recognition, natural language processing, and speech recognition."
        " Deep learning models, such as CNNs and RNNs, typically require large amounts of data and computational power."

)
I1003 = p.create_item(
    "I1003",
    R1__has_label="Feature",
    R2__has_description="In machine learning, features refer to measurable properties or characteristics extracted from input data."
     " For images, features may include patterns like edges and textures that help a model recognize objects, shapes, and more."
)

I1001 = p.create_item(
    "I1001",
    R1__has_label="Convolutional Neural Network",
    R2__has_description="A Convolutional Neural Network (CNN) is a deep learning algorithm primarily used for image processing."
    "CNNs apply convolution operations to detect patterns like edges, textures, and shapes in images, making them effective for tasks like image classification and object detection.",
    R4__is_instance_of=I1004["Deep Learning Model"],
    
    
)

I1002 = p.create_item(
    "I1002",
    R1__has_label="Image",
    R2__has_description="Images are visual representations in formats such as JPEG, PNG, or BMP."
    "They consist of pixels and are used as input for machine learning tasks like classification and object detection."
      "Image processing techniques help extract features from raw images."
)

I1002.set_relation(p.R000["contains"], I1003["Feature"])

# 'Image dataset' provide labeled 'samples' as 'training data' for 'supervised learning'.
I1008 = p.create_item(
    "I1008",
    R1__has_label="Training Data",
    R2__has_description="Training data is a set of data used to train a machine learning model."
        "It consists of input data and corresponding labels or outputs, which the model uses to learn patterns and relationships."
        "The quality and quantity of the training data significantly impact the performance of the trained model. Training data can be used in various machine learning tasks, including supervised and unsupervised learning."
)
I1005 = p.create_item(
    "I1005",
    R1__has_label="Image Dataset",
    R2__has_description="An image dataset is a collection of images used for training machine learning models."
      "It typically contains labeled data, where each image is associated with a label or category."
      "Image datasets are crucial for tasks like image classification, object detection, and segmentation, and they often require preprocessing like resizing and normalization before being used in training.",
        R15__is_element_of=I1008["Training Data"]
)

I1006 = p.create_item(
    "I1006",
    R1__has_label="Sample",
    R2__has_description="In machine learning, a sample refers to an individual data point or instance that is used in training or testing a model."
      "A sample can be an image, a piece of text, or any other type of data, and it typically represents one example of the data the model is learning to predict or classify.",
    R4__is_instance_of=I1008["Training Data"]
)

I1007 = p.create_item(
    "I1007",
    R1__has_label="Supervised Learning",
    R2__has_description="Supervised learning is a type of machine learning where the model is trained on a labeled dataset."
      "Each training sample consists of input data and the corresponding correct output, allowing the model to learn the relationship between the input and the output. It is commonly used for tasks like classification and regression."
    
)
I1005.set_relation(p.R000["is_input"], I1008["Training Data"])

I1005.set_relation(p.R000["contain"], I1006["Sample"])

# 'Loss functions' calculate the 'error' in predicting 'image labels'.
I1010 = p.create_item(
    "I1010",
    R1__has_label="Error",
    R2__has_description="In machine learning, error refers to the difference between the predicted output of a model and the actual output (ground truth). Errors can occur due to model misprediction, overfitting, or underfitting." 
    "Error analysis is crucial for improving the model and selecting the appropriate algorithm or training strategy."
)

I1009 = p.create_item(
    "I1009",
    R1__has_label="Loss Function",
    R2__has_description="A loss function is a mathematical function used to evaluate the performance of a machine learning model by quantifying the difference between the predicted output and the actual output (ground truth)."
      "The goal during training is to minimize the loss function, which helps the model improve its predictions over time. Common loss functions include Mean Squared Error (MSE) for regression tasks and Cross-Entropy Loss for classification tasks.",
    R16__has_property=I1010["Error"]
)


I1011 = p.create_item(
    "I1011",
    R1__has_label="Image Label",
    R2__has_description="An image label is a categorical annotation associated with an image in a dataset, typically used for supervised learning tasks." 
    "The label indicates the class or category to which the image belongs (e.g., 'cat,' 'dog,' or 'car'). Image labels are essential for training classification models, where the model learns to predict the correct label based on image features."
)
I1009.set_relation(p.R000["Evaluates"], I1011["Image Label"])


#  'Activation functions' introduce 'non-linearity' in 'deep neural networks'.

I1014 = p.create_item(
    "I1014",
    R1__has_label="Deep Neural Network",
    R2__has_description="A deep neural network (DNN) is a type of artificial neural network with multiple hidden layers between the input and output layers. These networks are capable of learning complex representations from data by utilizing multiple layers of non-linear transformations. DNNs are used for tasks like image classification, speech recognition, and language translation, where large datasets and complex patterns are involved."
)

I1013 = p.create_item(
    "I1013",
    R1__has_label="Non-Linearity",
    R2__has_description="Non-linearity in machine learning refers to the ability of a model to capture complex patterns and relationships in data that cannot be modeled as a simple linear equation."
      "It is introduced through activation functions in neural networks. Non-linearity allows models to approximate complex functions and is essential for tasks like image recognition and natural language processing.",
    R15__is_element_of=I1014["Deep Neural Network"]
)

I1012 = p.create_item(
    "I1012",
    R1__has_label="Activation Function",
    R2__has_description="An activation function is a mathematical function used in neural networks to introduce non-linearity into the model"
      "It determines whether a neuron should be activated or not by calculating a weighted sum of the inputs and then passing the result through the activation function. Common activation functions include ReLU (Rectified Linear Unit), Sigmoid, and Tanh, each influencing the learning and performance of the model.",
    R4__is_instance_of=I1015["Mathematical Function"],
    R16__has_property=I1013["Non-Linearity"]
)



# 'Softmax layers' produce 'probabilities' for different 'image classes'.


I1016 = p.create_item(
    "I1016",
    R1__has_label="Softmax Layer",
    R2__has_description="A Softmax layer is typically used as the final layer in a neural network for classification tasks." 
    "It converts the raw output scores of the network into probabilities by applying the Softmax function, which normalizes the scores so that they sum up to 1. Each value in the output represents the probability of a particular class, helping the model make a prediction based on the highest probability."
)

I1017 = p.create_item(
    "I1017",
    R1__has_label="Probability",
    R2__has_description="Probability is a measure of the likelihood that a given event will occur." 
    "In machine learning, probabilities are often used to represent the confidence of a model in its predictions. For example, in classification tasks, a model may output a probability distribution across different classes, with the highest probability indicating the most likely prediction."
)

I1018 = p.create_item(
    "I1018",
    R1__has_label="Image Class",
    R2__has_description="An image class is a label or category that an image belongs to in a classification task." 
    "For example, in an image classification model, the image class could represent categories such as 'cat', 'dog', or 'car'. The model is trained to predict the correct image class based on the features extracted from the image."
)


I1016.set_relation(p.R000["output"], I1017["Probability"])
I1016.set_relation(p.R000["assigns"], I1018["Image Class"])
I1017.set_relation(p.R000["Represent"], I1018["Image Class"])




# 'Training algorithms' 'optimize' and adjust model 'weights' using 'gradients'.
I1020 = p.create_item(
    "I1020",
    R1__has_label="Optimization Technique",
    R2__has_description="To optimize in machine learning refers to the process of adjusting the model parameters, such as weights and biases, to minimize the loss function."
      "Optimization techniques, like gradient descent, aim to find the optimal set of parameters that result in the best model performance by reducing the error over time."
)
I1021 = p.create_item(
    "I1021",
    R1__has_label="Weight",
    R2__has_description="In machine learning, a weight is a parameter within the model that is learned during training. Weights control the strength of the connection between neurons in a neural network."
      "During the training process, the weights are adjusted to minimize the error by optimizing the model’s performance based on the data and the loss function."
)
I1019 = p.create_item(
    "I1019",
    R1__has_label="Training Algorithm",
    R2__has_description="A training algorithm is a method used to adjust the parameters of a machine learning model in order to minimize the error or loss function" 
    "Common training algorithms include gradient descent, stochastic gradient descent, and Adam. These algorithms iteratively update the model’s parameters based on the data and the feedback (error) to improve its performance.",
    R4__is_instance_of=I1020["Optimization Technique"],
    R16__has_property=I1021["Weight"]
)


I1022 = p.create_item(
    "I1022",
    R1__has_label="Gradient",
    R2__has_description="A gradient refers to the derivative of a function with respect to its parameters."
      "In machine learning, gradients are used in optimization algorithms like gradient descent to update the model’s parameters"
      "The gradient indicates the direction and magnitude of change needed to reduce the error or loss function. The gradient is calculated during backpropagation in neural networks.",
    R15__is_element_of=I1020["Optimization Technique"]
)
I1019.set_relation(p.R000["Use"], I1022["Gradient"])


# **Dropout** deactivates **neurons**  to prevent **overfitting**, ensuring better generalization.

I1023 = p.create_item(
     "I1023",
     R1__has_label="Dropout",
     R2__has_description="A regularization technique that randomly deactivates neurons to reduce overfitting."
)
I1024 = p.create_item(
     "I1024",
     R1__has_label="Neurons",
     R2__has_description="Basic computational units in neural networks that process data and produce outputs."
 )

I1025 = p.create_item(
     "I1025",
     R1__has_label="Overfitting",
     R2__has_description="A condition where a model performs well on training data but poorly on unseen data."
 )
I1023.set_relation(p.R5["is part of"], I1024["Neurons"])

I1025.set_relation(p.R000["prevents"], I1025["Overfitting"])


# with open("machine_learning.svg", "w") as f:
#     f.write(p.visualize_entity(I1001.uri, radius = 3))
#pyirk --load-mod new_oml.py demo -vis __all__
#pyirk --load-mod oml.py demo -vis __all__

p.end_mod()