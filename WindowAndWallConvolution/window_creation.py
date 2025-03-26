import tensorflow as tf
from tensorflow.keras.layers import Layer

class WindowCreationLayer(Layer):
    def __init__(self,xdim_lower,xdim_upper,ydim_lower,ydim_upper,**kwargs):
        super().__init__(**kwargs)
        self.xdim_lower=xdim_lower
        self.xdim_upper=xdim_upper
        self.ydim_lower=ydim_lower
        self.ydim_upper=ydim_upper
        
    def call(self,input):
        out=input[:,self.xdim_lower:self.xdim_upper+1,self.ydim_lower:self.ydim_upper+1,:]
        return out
        
    def compute_output_shape(self,input_shape):
        return (input_shape[0],self.xdim_upper-self.xdim_lower+1,self.ydim_upper-self.ydim_lower+1,input_shape[3])
        
    def get_config(self):
        config=super().get_config()
        config.update(
            {
                "xdim_lower":self.xdim_lower,
                "xdim_upper":self.xdim_upper,
                "ydim_lower":self.ydim_lower,
                "ydim_upper":self.ydim_upper
            }
        )
        return config