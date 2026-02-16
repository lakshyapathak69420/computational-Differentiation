from tensor import Tensor
import math
class Tfs:

    @staticmethod
    def sine(x):
    
        if type(x) == Tensor:
            
            value = math.sin(x.value) 
            grad = x.grad*math.cos(x.value)
            return Tensor(value, grad)
        
        return math.sin(x)
    
    @staticmethod
    def log10(x):

        if type(x) == Tensor:
        
            value = math.log10(x.value) 
            grad = x.grad / x.value
            return Tensor(value, grad)
        
        return math.log10(x)
    
    @staticmethod
    def exp(x):
        if type(x) == Tensor:
            
            value = math.exp(x.value) 
            grad = x.grad * math.exp(x.value)
            return Tensor(value, grad)
        
        return math.exp(x)
    
    @staticmethod
    def cosine(x):
        if type(x) == Tensor:

            value = math.cos(x.value)
            grad = - x.grad * math.sin(x.value)
            return Tensor(value, grad)
        
        return math.cos(x)
    
    @staticmethod
    def tan(x):
        if type(x) == Tensor:
            value = math.tan(x.value)
            dx = math.cos(x.value)

            grad = x.grad / dx
            return Tensor(value, grad)
        
        return math.tan(x)





