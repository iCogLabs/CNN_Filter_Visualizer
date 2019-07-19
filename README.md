# Convolutional Neural Network Visualizations 

This module contains a Guided BackPropagation convolutional neural network visualization techniques implemented in PyTorch.

how to use:
    after importing the module into your program
    1 - create an object of the GuidedBackprop class by 
        passing a model with items the are itrable

    2 - Then using the visualize functions in the same class
        you can visualze each fiter in each layer
        by passing the appropriate parameters

gbp = GuidedBackprop(model)
gbp(PIL_Image, layer_index, filter_index, normalize=True)

then the gradients will be ploted 
    