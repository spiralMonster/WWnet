import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../')))

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Activation,BatchNormalization,Concatenate,Conv2D
from RandomizedWindowAndWallConvolution.randomized_window_wall_convolution import RandomizedWindowAndWallConvolutionLayer

class SkipConnectionLayer(Layer):
    def __init__(self,layer_type,layer_config,**kwargs):
        super().__init__(**kwargs)
        self.layer_type=layer_type
        self.layer_config=layer_config
        self.layer=None
        
        #Initializing layer based upon layer type:
        if self.layer_type=="conv":
            self.layer=Conv2D(
                filters=self.layer_config["filters"],
                kernel_size=self.layer_config["kernel_size"],
                activation=None,
                kernel_initializer=self.layer_config["kernel_initializer"],
                padding=self.layer_config["padding"]
            )
            self.activation=Activation(activation=self.layer_config["activation"])
            self.out_depth=self.layer_config["filters"]
            
        if self.layer_type=="random_window_wall_conv":
            self.layer=RandomizedWindowAndWallConvolutionLayer(
                number_window_wall_instances=self.layer_config["number_window_wall_instances"],
                window_wall_layer_config=self.layer_config["window_wall_layer_config"]
            )
            self.activation=self.layer_config["window_wall_layer_config"]["window_layer_config"]["activation"]
            
            self.out_depth=self.layer_config["window_wall_layer_config"]["window_layer_config"]["filters"]*self.layer_config["number_window_wall_instances"]
            
    def build(self,input_shape):
        #Building layer:
        self.layer.build(input_shape)
        
        super().build(input_shape)
        
    def call(self,inputs):
        x=inputs
        x=self.layer(x)
        x=BatchNormalization()(x)
        x=Activation(activation=self.activation)(x)
        
        out=Concatenate(axis=-1)([inputs,x])
        return out
        
    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[1],input_shape[2],input_shape[3]+self.out_depth)
        
    def get_config(self):
        config=super().get_config()
        config.update({
            "layer_type":self.layer_type,
            "layer_config":self.layer_config
        })
        return config


        
    
        
        
