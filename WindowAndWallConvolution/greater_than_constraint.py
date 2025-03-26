import tensorflow as tf
from tensorflow.keras.constraints import Constraint

class GreaterThanConstraint(Constraint):
    def __init__(self,lower,min_margin=1,**kwargs):
        super().__init__(**kwargs)
        self.lower=lower
        self.min_margin=min_margin
        
    def __call__(self,upper):
        return tf.maximum(upper,self.lower+self.min_margin)
         
    def get_config(self):
        config=super().get_config()
        config.update(
            {
                "lower":self.lower,
                "min_margin":self.min_margin
            }
        )
        return config