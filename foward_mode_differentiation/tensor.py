class Tensor:
    """
    a + b*ep is called a grad number (called it a tensor here) iff following is true
    - ep^2 = 0
    - ep != 0
    it is pretty similar to complex number
    """
    def __init__(self, value, grad = 1):
        self.value = value
        self.grad = grad

    def __add__(self, other):
        if type(other) != Tensor:
            value = self.value + other
            return Tensor(value, self.grad) 
        
        grad = self.grad + other.grad
        value = self.value + other.value
        return Tensor(value , grad)
    
    def __sub__(self, other):
        if type(other) != Tensor:
            value = self.value - other
            return Tensor(value, self.grad) 
        
        grad = self.grad - other.grad
        value = self.value - other.value
        return Tensor(value, grad)

    def __mul__(self, other):
        if type(other) != Tensor:
            self.value *= other
            self.grad *= other
            return self 
        x = self.value
        x_p = self.grad
        y = other.value
        y_p = other.grad

        value = x*y
        grad = x*y_p + y*x_p
        return Tensor(value , grad)
    
    def __truediv__(self, other):
        if type(other) != Tensor:
            value = self.value / other
            grad = self.grad / other
            return Tensor(value, grad) 
        
        x = self.value
        x_p = self.grad
        y = other.value
        y_p = other.grad
        value = x / y
        grad = (y*x_p - x*y_p) / (y**2)
        return Tensor(value , grad)
    
    def __pow__(self, n):
        a = self.value
        b = self.grad
        c = n-1 
        grad = n * (a**c) * b
        value = self.value ** n
        return Tensor(value, grad)
    
    def printf(self):
        print(f"{self.value}")

    def get_gradient(self):
        return self.grad

