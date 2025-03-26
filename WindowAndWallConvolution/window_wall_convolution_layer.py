import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"./")))

import tensorflow as tf
from tensorflow.keras.layers import Layer,Conv2D
from greater_than_constraint import GreaterThanConstraint
from window_creation import WindowCreationLayer
from wall_creation import WallCreationLayer
from combining_window_wall import CombiningWindowAndWallLayer

class WindowAndWallConvolutionLayer(Layer):
    def __init__(self,window_layer_config,wall_layer_config,**kwargs):
        super().__init__(**kwargs)
        self.window_layer_config=window_layer_config
        self.wall_layer_config=wall_layer_config
       
        # Initializing window_convolution_layer:
        self.window_convolution_layer=Conv2D(
           filters=self.window_layer_config["filters"],
           kernel_size=self.window_layer_config["kernel_size"],
           activation=self.window_layer_config["activation"],
           kernel_initializer=self.window_layer_config["kernel_initializer"],
           padding=self.window_layer_config["padding"]
         
        )
        
        #Initializing wall_convolution_layer:
        self.wall_convolution_layer=Conv2D(
           filters=self.wall_layer_config["filters"],
           kernel_size=self.wall_layer_config["kernel_size"],
           activation=self.wall_layer_config["activation"],
           kernel_initializer=self.wall_layer_config["kernel_initializer"],
           padding=self.wall_layer_config["padding"]
         
        )
        
    def build(self,input_shape):
        
        # X dimensions:
        
        self.xlower=self.add_weight(
            shape=(),
            initializer=tf.random.uniform(shape=(),
                                          minval=0,
                                          maxval=input_shape[1]-1),
            trainable=True,
            name="x_lower_dim",
            dtype=tf.float32
        )
        
        self.xupper=self.add_weight(
            shape=(),
            initializer=tf.random.uniform(shape=(),
                                          minval=self.xlower+1,
                                          maxval=input_shape[1]-1),
            trainable=True,
            constraint=GreaterThanConstraint(lower=self.xlower),
            name="x_upper_dim",
            dtype=tf.float32
          
        )
        
        # Y dimensions:
        self.ylower=self.add_weight(
            shape=(),
            initializer=tf.random.uniform(shape=(),minval=0,maxval=input_shape[2]-1),
            trainable=True,
            name="y_lower_dim",
            dtype=tf.float32
        )
        
        self.yupper=self.add_weight(
            shape=(),
            initializer=tf.random.uniform(shape=(),minval=self.ylower+1,maxval=input_shape[2]-1),
            trainable=True,
            constraint=GreaterThanConstraint(lower=self.ylower),
            name="y_upper_dim",
            dtype=tf.float32
          
        )
        
        #Building Window convolution Layer:
        self.window_convolution_layer.build(input_shape=(
            input_shape[0],
            tf.cast(self.xupper,dtype=tf.int32).numpy()-tf.cast(self.xlower,dtype=tf.int32).numpy()+1,
            tf.cast(self.yupper,dtype=tf.int32).numpy()-tf.cast(self.ylower,dtype=tf.int32).numpy()+1, 
            input_shape[3]
        ))
        
        #Building Wall convolution Layer:
        self.wall_convolution_layer.build(input_shape)
    
        super().build(input_shape)
        
       

    
    def call(self,input):
        window=WindowCreationLayer(
            xdim_lower=tf.cast(self.xlower,dtype=tf.int32).numpy(),
            xdim_upper=tf.cast(self.xupper,dtype=tf.int32).numpy(),
            ydim_lower=tf.cast(self.ylower,dtype=tf.int32).numpy(),
            ydim_upper=tf.cast(self.yupper,dtype=tf.int32).numpy()
        )(input)
       
        #Creating Wall:
        window_dimensions={
            "x":{
                "lower":tf.cast(self.xlower,dtype=tf.int32).numpy(),
                "upper":tf.cast(self.xupper,dtype=tf.int32).numpy()
            },
            "y":{
                "lower":tf.cast(self.ylower,dtype=tf.int32).numpy(),
                "upper":tf.cast(self.yupper,dtype=tf.int32).numpy()
            }
        }
        
        wall=WallCreationLayer(
            window_dimensions=window_dimensions
        )(input)
       
        #Convolution for window:
        window_conv=self.window_convolution_layer(window)
        
        
        #Convolution for wall:
        wall_conv=self.wall_convolution_layer(wall)
        
        
        # Combining window and wall:
        out=CombiningWindowAndWallLayer(
            window_dimensions=window_dimensions
        )(wall=wall_conv,window=window_conv)
       
        return out
        
    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[1],input_shape[2],self.wall_layer_config["filters"])
        
    def get_config(self):
        super().get_config()
        config.update(
            {
                "window_layer_config":self.window_layer_config,
                "wall_layer_config":self.wall_layer_config
            }
        )
        return config
        



        