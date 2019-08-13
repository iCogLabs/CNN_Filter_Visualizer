# Convolutional Neural Network Visualizations 

This module contains a convolutional neural network visualization 
techniques implemented using Guided BackPropagation in PyTorch.

The implementation here is based on the repo from 
## Requierments

1. pytorch
2. numpy
3. matplotlib
4. PIL


# Installation:

## To manually install from this repository:

1. First make sure you have the required packages above
2. Clone the repository
3. Navigate in to the cloned folder and then to the src subdirectory
4. Then run:
        -$ python setup.py install


## To install using pip:
### Run the following:

    pip install CNNFilterVisualizer

## To test your installation:
open the Python REPL and import CNNFilterVisualizer

# How to use:
    
After importing the module into your program

1. Create an object of the FilterVisualizer class by 
    passing a model with "Sequential" layers

2. Then using the visualize functions in the same class
    you can visualze each fiter in each layer
    by passing the appropriate parameters


```python
    # first create an object of the class by passing your model
    gbp = FilterVisualizer(model=your_model)

    # call the visualize method passing in the image, layer_index and filter_index
    # note: to visualize all the filters in a layer, pass in the string "All" to the 
    # parameter filter_index
    gbp.visualize(input_image=image_used_for_visualization, 
                cnn_layer=model_layer_to_visualize, 
                filter_pos=filter_of_the_layer_to_visualize, 
                normalize=True)


```

### That's it. If you run this sucessfully you will see the filter activations of the layer you chose.

You can also checkout the example notebook in this repo to see the activations features of an alexnet model with a picture of a dog.

## Here are some sample outputs from the 2nd convolutiona layer in a vgg model

### Input image :
![Input](./images/dog.jpg)


### Activations :
![Filter Visualizations](./images/1.png)

## Refernece :
@utkuozbulak pytorch-cnn-visualizations repo

If you want to understand more about the implemtation i recomend that you check out the above repo by @utkuozbulak . It has a lot more detail examples and also some other visualization techniques you can checkout.

    