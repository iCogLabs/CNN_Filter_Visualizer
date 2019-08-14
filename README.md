# Convolutional Neural Network Visualizations 

This module contains a convolutional neural network visualization 
techniques implemented using Guided BackPropagation in Tensorflow.
Even if the implementation is in Tensorflow the module inculdes a 
model convertion class that enables it to support multiple other frameworks, such as, 
Pytorch, Caffe and Theano.

The implementation here is based on the repo from [here](https://github.com/conan7882/CNN-Visualization)
## Requierments

1. pytorch
2. Tensorflow 
2. numpy
3. matplotlib
4. PIL
5. onnx
6. onnx-tf


# Installation:

## To manually install from this repository:

1. First make sure you have the required packages above
2. Clone the repository
3. Navigate in to the cloned folder and then to the src subdirectory. Then move into the 'main' folder
4. Then run:
        -$ python setup.py install


## To install using pip:
### Run the following:

    pip install CNNFilterVisualizer

## To test your installation:
open the Python REPL and import CNNFilterVisualizer

# How to use:
    
After importing the module into your program

1. Create an object of the **FilterVisualizer** class by 
    passing a pytorch or a tensorflow model

2. Then using the **visualize** functions in the same class
    you can visualze each fiter in each layer
    by passing the appropriate parameters


```python
    # first create an object of the class by passing your model
    fv = FilterVisualizer(model=your_model, framework='tensorflow')

    # call the visualize method passing in the image, layer_index and filter_index
    # note: to visualize all the filters in a layer, pass in the string "All" to the 
    # parameter filter_index
    fv.visualize(input_image=image_used_for_visualization, 
                cnn_layer=model_layer_to_visualize, 
                filter_pos=filter_of_the_layer_to_visualize, 
                normalize=bool_value_that_determines_normalization)


```

### That's it. If you run this sucessfully you will see the filter activations of the layer you chose.
## It might take a few seconds to plot the activations

You can also checkout the example notebook in this repo to see the activations features of an alexnet model with a picture of a dog.

## Here are some sample outputs from the 2nd convolutiona layer in a vgg model

### Input image :
![Input](./images/dog.jpg)


### Activations :
![Filter Visualizations](./images/1.png)

## Refernece :
1. [Striving for Simplicity: The All Convolutional Net](https://arxiv.org/abs/1412.6806)
2. @utkuozbulak pytorch-cnn-visualizations [repo](https://github.com/utkuozbulak/pytorch-cnn-visualizations)
3. @conan7882 [repo](https://github.com/conan7882/CNN-Visualization)

If you want to understand more about the implemtation i recomend that you check out the above references.

    