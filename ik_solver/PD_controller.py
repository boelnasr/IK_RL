class PDController:
    def __init__(self, kp=1.0, kd=0.1, dt=0.01):
        """
        Initialize PD controller.
        
        Args:
            kp (float): Proportional gain
            kd (float): Derivative gain
            dt (float): Time step for derivative calculation
        """
        self.kp = kp
        self.kd = kd
        self.dt = dt
        self.previous_error = None
        
    def compute(self, error):
        """
        Compute PD control signal.
        
        Args:
            error (float): Current error value
            
        Returns:
            float: Control signal
        """
        if self.previous_error is None:
            self.previous_error = error
            derivative = 0
        else:
            derivative = (error - self.previous_error) / self.dt
        
        control = self.kp * error + self.kd * derivative
        self.previous_error = error
        return control
        
    def reset(self):
        """Reset the controller state."""
        self.previous_error = None