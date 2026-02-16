from tensor import Tensor
import numpy as np
class tensorFn:
    @staticmethod
    def relu(tensor : Tensor):
        operation = "relu"
        if not isinstance(tensor, Tensor):
            raise TypeError("Not a Tensor")
        
        value = np.maximum(0, tensor.value)
            
        return Tensor(value, operation, tensor)
    
    @staticmethod   
    def tanh(tensor: Tensor):
        operation = "tanh"
        if not isinstance(tensor, Tensor):
            raise TypeError("Not a Tensor") 
        
        value = np.tanh(tensor.value)
        return Tensor(value, operation, tensor)
    
    @staticmethod
    def matmul(t1 : Tensor, t2 : Tensor):
        operation = "matmul"
        if not isinstance(t1, Tensor):
            raise TypeError(f"t1 is Not a Tensor") 
        
        if not isinstance(t2, Tensor):
            raise TypeError("t2 is Not a Tensor") 
        
        try:
            value = np.matmul(t1.value, t2.value)
        except ValueError:
            raise ValueError

        return Tensor(value, operation, t1, t2)
    
    @staticmethod
    def cross_entropy_loss(logit: Tensor, label_index: int):
    
        if not isinstance(logit, Tensor):
            raise TypeError("logits need to be type of tensor")
        
        
        one_hot_array = np.zeros_like(logit.value, dtype=np.float32)
        one_hot_array[0][label_index] = 1.0
        original_label = Tensor(one_hot_array)
        
        exp_values = np.exp(logit.value)
        sum_exp = np.sum(exp_values)
        
        correct_class_exp = exp_values[0][label_index]
        
        probability = correct_class_exp / sum_exp
        loss_value = -np.log(probability + 1e-9)
        return Tensor(loss_value, "CE", logit, original_label)

