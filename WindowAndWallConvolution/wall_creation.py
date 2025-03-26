import tensorflow as tf
from tensorflow.keras.layers import Layer

# window_dimensions={
#     "x":{
#         "lower":,
#         "upper":
         
#     },
#    "y":{
#        "lower":,
#        "upper":
#    }
# }

class WallCreationLayer(Layer):
    def __init__(self,window_dimensions,**kwargs):
        super().__init__(**kwargs)
        self.window_dimensions=window_dimensions
        
    def call(self,input):
        window_x_lower=self.window_dimensions["x"]["lower"]
        window_x_upper=self.window_dimensions["x"]["upper"]
        
        window_y_lower=self.window_dimensions["y"]["lower"]
        window_y_upper=self.window_dimensions["y"]["upper"]

        out=input

        target_square=tf.zeros_like(out[:,window_x_lower:window_x_upper+1,window_y_lower:window_y_upper+1,:])
        target_out_above=out[:,window_x_lower:window_x_upper+1,:window_y_lower,:]
        target_out_below=out[:,window_x_lower:window_x_upper+1,window_y_upper+1:,:]
        target_layer=tf.concat([target_out_above,target_square,target_out_below],axis=2)

        layer_before=out[:,:window_x_lower,:,:]
        layer_after=out[:,window_x_upper+1:,:,:]
        
        final_out=tf.concat([layer_before,target_layer,layer_after],axis=1)
        
        
        return final_out
        
    def compute_output_shape(self,input_shape):
        return input_shape
        
    def get_config(self):
        config=super().get_config()
        config.update(
            {
                "window_dimensions":self.window_dimensions
            }
        )
        return config
        
        