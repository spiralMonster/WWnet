import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Concatenate
from WindowAndWallConvolution.window_wall_convolution_layer import WindowAndWallConvolutionLayer

# window_wall_layer_config={
#     "window_layer_config":{
        
#     },
#     "wall_layer_config":{
        
#     }
# }

class RandomizedWindowAndWallConvolutionLayer(Layer):
    def __init__(self,number_window_wall_instances,window_wall_layer_config,**kwargs):
        super().__init__(**kwargs)
        self.number_window_wall_instances=number_window_wall_instances
        self.window_wall_layer_config=window_wall_layer_config
        
        #Initializing WindowAndWallConvolution Layers:
        self.window_wall_conv_layers=[]
        
        for _ in range(self.number_window_wall_instances):
            layer=WindowAndWallConvolutionLayer(
                window_layer_config=self.window_wall_layer_config["window_layer_config"],
                wall_layer_config=self.window_wall_layer_config["wall_layer_config"]
            )
            self.window_wall_conv_layers.append(layer)
            
        self.stack=[]

    
    def build(self,input_shape):
        #Building WindowAndWallConvolution Layers:
        for layer in self.window_wall_conv_layers:
            layer.build(input_shape)
            
        super().build(input_shape)
            
    
    def call(self,inputs):
        # WindowAndWallConvolutionLayer:
        for ind in range(self.number_window_wall_instances):
            out=self.window_wall_conv_layers[ind](inputs)
            self.stack.append(out)
            
        #Stacking window_wall_conv_layers:
        final_out=Concatenate(axis=-1)(self.stack)
        return final_out
        
    def compute_output_shape(self,input_shape):
        return (input_shape[0],
                input_shape[1],
                input_shape[2],
                self.number_window_wall_instances*self.window_wall_layer_config["window_layer_config"]["filters"])
        
    def get_config(self):
        config=super().get_config()
        config.update(
            {
                "number_window_wall_instances":self.number_window_wall_instances,
                "window_wall_layer_config":self.window_wall_layer_config
            }
        )
        return config
        
            
            


        