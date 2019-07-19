"""
Created on Fri Jul 19 10:26:38 2019

@author: henok
"""

# Import the neccessary modules
import torch
from torch.nn import ReLU
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt

class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.gradients_to_visualize = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Register hook to the first layer
        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for pos, module in self.model.features._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def preprocess_image(self, pil_im, resize_im=True):
        """
            Processes image for CNNs

        Args:
            PIL_img (PIL_img): Image to process
            resize_im (bool): Resize to 224 or not
        returns:
            im_as_var (torch variable): Variable that contains processed float tensor
        """
        # mean and std list for channels (Imagenet)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # Resize image
        if resize_im:
            pil_im.thumbnail((512, 512))
        im_as_arr = np.float32(pil_im)
        im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
        # Normalize the channels
        for channel, _ in enumerate(im_as_arr):
            im_as_arr[channel] /= 255
            im_as_arr[channel] -= mean[channel]
            im_as_arr[channel] /= std[channel]
        # Convert to float tensor
        im_as_ten = torch.from_numpy(im_as_arr).float()
        # Add one more channel to the beginning. Tensor shape = 1,3,224,224
        im_as_ten.unsqueeze_(0)
        # Convert to Pytorch variable
        im_as_var = Variable(im_as_ten, requires_grad=True)
        return im_as_var


    def visualize(self, input_image, cnn_layer, filter_pos, normalize=True):
        """
            Generate and Visualize gradients from the image to visualize the filter activations

        Args:
            input_image (PIL_img): Image to process
            cnn_layer (int): The layer index to visualize
            filter_pos (int): The filter in the chosen layer to visualize
            normalize (bool): choose if normalization should be done
        returns:
            gradients_as_arr (numpy.ndarray): numpy array that contains the gradients
        """
        # First Preprocess the image 
        input_image = preprocess_image(input_image)
        self.model.zero_grad()
        # Forward pass
        x = input_image
        for index, layer in enumerate(self.model.features):
            # Forward pass layer by layer
            # x is not used after this point because it is only needed to trigger
            # the forward hook function
            x = layer(x)
            # Only need to forward until the selected layer is reached
            if index == cnn_layer:
                # (forward hook function triggered)
                break
        conv_output = torch.sum(torch.abs(x[0, filter_pos]))
        # Backward pass
        conv_output.backward()
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]

        if normalize:
            gradients_as_arr -= gradients_as_arr.min()
            gradients_as_arr /= gradients_as_arr.max()
            gradients_as_arr = np.uint8(gradients_as_arr * 255)
        
        # save the gradients of the selected filter in a class variable "gradients_to_visualize"
        # for optional access to the gradients
        self.gradients_to_visualize = gradients_as_arr
        
        # use plt.imshow to plot the gradients
        plt.imshow(np.transpose(gradients_as_arr, (1, 2, 0)))
        
        return "Done"

