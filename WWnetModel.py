import tensorflow as tf
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,Concatenate,Activation,BatchNormalization,Dense,Flatten
from tensorflow.keras.models import Model
from skip_connection import SkipConnectionLayer


# For SkipConnectionLayer:
# if layer_type=="conv":
#     layer_config={
#         "filters":,
#         "kernel_size":,
#         "activation":,
#         "kernel_initializer":,
#         "padding":
#     }


# if layer_type=="random_window_wall_conv":

#     window_wall_layer_config={
#     "window_layer_config":{
#         "filters":,
#         "kernel_size":,
#         "activation":,
#         "kernel_initializer":,
#         "padding":
        
#     },
#     "wall_layer_config":{
#         "filters":,
#         "kernel_size":,
#         "activation":,
#         "kernel_initializer":,
#         "padding":
#     }
# }

#     layer_config={
#         "number_window_wall_instances":,
#          "window_wall_layer_config":window_wall_layer_config
#     }


class WWnetModel:
    def __init__(self,n_model_units,wwnet_config,random_window_wall_conv_layer_config,conv_layer_config,dense_layer_config,last_layer_config):
        self.n_model_units=n_model_units
        self.wwnet_config=wwnet_config
        self.random_window_wall_conv_layer_config=random_window_wall_conv_layer_config
        self.conv_layer_config=conv_layer_config
        self.dense_layer_config=dense_layer_config
        self.last_layer_config=last_layer_config
        
    def build_model(self,input_shape):
        inp=Input(shape=input_shape,dtype=tf.float32)
        rwwc_out=inp
        cnn_out=inp
        wwnet_out=inp
        
        # Creating Model:
        for ind in range(self.n_model_units):
            # Going through Normal CNN
            cnn_out=SkipConnectionLayer(layer_type="conv",
                                        layer_config=self.conv_layer_config[ind])(cnn_out)
            
            #Going through Randomized Window and Wall Convolution:
            rwwc_out=SkipConnectionLayer(layer_type="random_window_wall_conv",
                                         layer_config=self.random_window_wall_conv_layer_config[ind])(rwwc_out)
            
            #Concatinating CNN and RWCC for input to WWnet:
            wwnet_out=Concatenate(axis=-1)([wwnet_out,cnn_out,rwwc_out])
            
            #Going through WWnet:
            config=self.wwnet_config[ind]
            activation=config["activation"]
            
            wwnet_out=Conv2D(
                filters=config["filters"],
                kernel_size=config["kernel_size"],
                activation=None,
                kernel_initializer=config["kernel_initializer"],
                padding=config["padding"]
            )(wwnet_out)

            wwnet_out=BatchNormalization()(wwnet_out)
            wwnet_out=Activation(activation)(wwnet_out)

            #MaxPooling WWnetLayer:
            wwnet_out=MaxPooling2D((2,2),padding="same")(wwnet_out)

            #MaxPooling CNN and RWCC layers:
            cnn_out=MaxPooling2D((2,2),padding="same")(cnn_out)
            rwwc_out=MaxPooling2D((2,2),padding="same")(rwwc_out)
            
        wwnet_out=Flatten()(wwnet_out)
        
        #Dense Layer architecture:
        for config in self.dense_layer_config:
            wwnet_out=Dense(
                units=config["units"],
                activation=config["activation"],
                kernel_initializer=config["kernel_initializer"],
                kernel_regularizer=config["kernel_regularizer"]
            )(wwnet_out)
            
        #Final Layer:
        wwnet_out=Dense(
             units=self.last_layer_config["units"],
             activation=self.last_layer_config["activation"],
             kernel_initializer=self.last_layer_config["kernel_initializer"],
             kernel_regularizer=self.last_layer_config["kernel_regularizer"]
        )(wwnet_out)
        
        #Model Initialization:
        model=Model(inputs=inp,outputs=wwnet_out)
        return model
            
        


    
        
        
    
    