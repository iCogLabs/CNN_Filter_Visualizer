import numpy as np
import tensorflow as tf
import torch
import keras
from keras import backend as K
import onnx
import onnx_tf
from onnx_tf.backend import prepare
from collections import OrderedDict
import os
import matplotlib.pyplot as plt


class pytorch_to_tensorflow_converter():
    def __init__(self, pytorch_model, output_layer, input_dims):
        self.__model = pytorch_model
        self.__output_layer = output_layer
        self.__input_dims = input_dims
        self.__new_model = None
        self.output_tensor = None
        self.input_tensor = None
        self.tf_graph = None
        self._slice_model()
        self._convert_to_tf()
        
        
    
    def _slice_model(self):
        module = []
        output_layer = self.__output_layer
        counter = 0

        # Replace with recursive
        for _, layer in enumerate(self.__model.children()):
            if len(list(layer.children()))==0:
                module.append(layer)
                counter += 1
                if counter == output_layer:
                    break
            else:
                for _, inner_layer in enumerate(layer.children()):
                    if len(list(inner_layer.children()))==0:
                        module.append(inner_layer)
                        counter += 1
                        if counter == output_layer:
                            break
                    else:
                        for _, inner_layer in enumerate(layer.children()):
                            if len(list(inner_layer.children()))!=0:
                                module.append(inner_layer)
                                counter += 1
                                if counter == output_layer:
                                    break
                    if counter == output_layer:
                        break
            if counter == output_layer:
                    break
                    
        self.__new_model = torch.nn.Sequential(*module)

        

    def _convert_to_tf(self):
        onnx_output_dir = './onnx_output/model.onnx'
        
        if not os.path.exists('./onnx_output/'):
            os.makedirs('./onnx_output/')
        if not os.path.exists('./tf_model/'):
            os.makedirs('./tf_model/')

        dummy_input = torch.rand(self.__input_dims)

        # Export the onnx representation to the onnx_output_dir
        torch.onnx.export(self.__new_model, dummy_input, 
                          onnx_output_dir, input_names=['input'], 
                          output_names=['output'])

        # Load ONNX representation and convert to TensorFlow format
        model_onnx = onnx.load(onnx_output_dir)
        tf_rep = prepare(model_onnx, strict=False)
        
        # Get the output and input tensors from the tf_rep
        with tf.Session() as sess:
            sess.graph.as_default()
            tf.import_graph_def(tf_rep.graph.as_graph_def(), name='')
            
            self.input_tensor = sess.graph.get_tensor_by_name(
                tf_rep.tensor_dict['input'].name
            )
            self.output_tensor = sess.graph.get_tensor_by_name(
                tf_rep.tensor_dict['output'].name
            )
        
            self.tf_graph = sess.graph



@tf.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(self, op, grad):
    return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), 
                    tf.zeros(grad.get_shape()))



class FilterVisualizer():
    def __init__(self, model, framework='Tensorflow'):
        self.model = model
        self.framework = framework
        if self.framework.lower() == "tensorflow":
            self.model.trainable = False
        self.gradients = None
    

    def _show_all(self, images, cols = 3, titles = None, normalize=False):
        """Display a list of images in a single figure with matplotlib.

        Parameters
        ---------
        images: List of np.arrays compatible with plt.imshow.

        cols (Default = 1): Number of columns in figure (number of rows is 
                            set to np.ceil(n_images/float(cols))).

        titles: List of titles corresponding to each image. Must have
                the same length as titles.
        """
        assert((titles is None)or (len(images) == len(titles)))
        n_images = len(images)
        titles = ['Filter (%d)' % i for i in titles]
        fig = plt.figure()
        
        if n_images > 1:
            for n, (image, title) in enumerate(zip(images, titles)):
                a = fig.add_subplot(np.ceil(n_images/float(cols)), cols, n + 1)
                if image[0].ndim == 2:
                    plt.gray()
                
                if normalize:
                    image -= image.min()
                    image /= image.max()
                    image = np.uint8(image * 255)
            
                plt.imshow(image[0])
                a.set_title(title, fontsize=(fig.get_size_inches()[0]*n_images))
            if n_images<3:
                multiplier = 3
            else:
                multiplier = n_images/2
            fig.set_size_inches(np.array(fig.get_size_inches()) * (n_images, cols*multiplier) )
    
            plt.show()
        
        else:
            if normalize:
                images[0] -= images[0].min()
                images[0] /= images[0].max()
                images[0] = np.uint8(images[0] * 255)
            
            plt.imshow(images[0][0])
            plt.title(titles[0])
            
    
    
    def visualize(self, input_image, cnn_layer, filters, normalize=True):
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
        if not isinstance(filters, list):
            filters = [filters, filters+1]
            
        # Here we are defining the intermidiate layer outputs we want
        if self.framework.lower() == "pytorch":
            # Convert the pytorch model to a tf graph
            converted_model = pytorch_to_tensorflow_converter(self.model, cnn_layer, input_image.shape)
            
            # Get the input and output tensor and the tf graph for inference
            input_tensor = converted_model.input_tensor
            output_tensor = converted_model.output_tensor
            g = converted_model.tf_graph
            
            # Run the input image through the converted graph
            # to get the ouput of the layer we wanna visualize
            sess = tf.Session(graph = g)
            intermediate_output = sess.run(output_tensor, feed_dict={input_tensor: input_image})
            # plt.imshow(intermediate_output[0][0])
            
            layer_output = output_tensor
            
            with g.gradient_override_map({'Relu': 'GuidedRelu'}):
                outputs = layer_output[:,filters[0]:filters[-1],:,:]                
                indexes = outputs.shape[1]

                with sess:
                    # Calculate and stack the gradients for all the filters
                    grads = tf.stack([tf.gradients(ys=[outputs[:,i,:,:]][0] , 
                                                   xs=[input_tensor])[0] 
                                      for i in range(indexes)], axis=1)
    
                    var_grad_vals = K.function([input_tensor], [grads])
                    var_grad_vals = var_grad_vals([input_image])[0][0]

                    # Use Transopse to change the format to (n, h, w, c) 
                    # this i because tf and pytorch use diffrent image formats
                    # (channel firs and channel last)
                    var_grad_vals = np.transpose(var_grad_vals, (0, 2, 3, 1))

                    # Do a split to pass the appropriate format for the plotting function.
                    var_grad_vals = np.array_split(var_grad_vals, var_grad_vals.shape[0], axis=0)
                    # save them as a class field
                    self.gradients = var_grad_vals

        else:
            input_node = self.model.input
            intermediate_output = self.model.layers[cnn_layer].output
        
            get_intermediate_output = K.function([input_node,
                                                  K.learning_phase()],
                                                 [intermediate_output])
            
            # Assign a handler for the graph
            g = tf.get_default_graph()
            
            # outputs in test mode = 0
            layer_output = intermediate_output
        
            with g.gradient_override_map({'Relu': 'GuidedRelu'}):
                outputs = layer_output[:,:,:,filters[0]:filters[-1]]
                indexes = outputs.shape[-1]

                grads = tf.stack([tf.gradients(ys=[outputs[:,:,:,i]][0] , xs=[input_node])[0] for i in range(indexes)], axis=1)
   
                var_grad_vals = K.function([input_node], [grads])
                var_grad_vals = var_grad_vals([input_image])[0][0]

                # Do a split to pass the appropriate format for the plotting function.
                var_grad_vals = np.array_split(var_grad_vals, var_grad_vals.shape[0], axis=0)
                # save them as a class field
                self.gradients = var_grad_vals
            
        self._show_all(images=var_grad_vals, 
                        titles=range(filters[0], filters[-1]), 
                       normalize=normalize)