# Convolutional Neural Network Visualizations 

This module contains a convolutional neural network visualization 
techniques implemented using Guided BackPropagation in PyTorch.

the implementation here is based on the repo from 
## Requierments

1. pytorch
2. numpy
3. matplotlib
4. PIL


# Installation:

## To manually install from this repository:

1. first make sure you have the required packages above
2. clone the repository
3. Navigate in to the cloned folder
4. run python setup.py install


## To install using pip:
### Run the following:
    
    pip install CNNFilterVisualizer

## To test your installation:
    open the Python REPL and import CNNFilterVisualizer

# How to use:
    
after importing the module into your program

1. create an object of the FilterVisualizer class by 
    passing a model with "Sequential" layers

2. Then using the visualize functions in the same class
    you can visualze each fiter in each layer
    by passing the appropriate parameters


```python
    # first create an object of the class by passing your model
    gbp = GuidedBackprop(model=your_model)

    # call the visualize method passing in the image, layer_index and filter_index
    # note: to visualize all the filters in a layer, pass in the string "All" to the 
    # parameter filter_index
    gbp.visualize(PIL_Image=image_used_for_visualization, 
                layer_index=model_layer_to_visualize, 
                filter_index=filter_of_the_layer_to_visualize, 
                normalize=True)

```

### That's it. If you run this sucessfully you will see the filter activations of the layer you chose.

You can also checkout the example notebook in this repo to see the activations features of an alexnet model with a picture of a dog.

## Here are some sample outputs from the earlier layer in a vgg model
![Filter Visualizations](./images/1.png)

## Refernece :
@utkuozbulak pytorch-cnn-visualizations repo

if you want to understand more about the implemtation i recomend that you check out the above repo by @utkuozbulak . It has a lot more detail examples and also some other visualization techniques you can checkout.

    