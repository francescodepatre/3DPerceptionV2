import numpy as np

class KalmanFilter:
    def __init__(self, F, B, H, Q, R, x0, P0):
        self.F = F
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0

    def predict(self, u):
        #Predizione sullo stato e la sua covarianza
        self.x = np.dot(self.F,self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x,self.P
    
    def update(self, z):
        #calcola il guadagno del filtro di kalman 
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))      
        # Update the state estimate and covariance matrix
        y = z - np.dot(self.H, self.x)  
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.P.shape[0])
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)
        return self.x, self.P
# Example usage

def test():
    F = np.array([[1, 1], [0, 1]]) 
    B = np.array([[0.5], [1]])     
    H = np.array([[1, 0]])         
    Q = np.array([[1, 0], [0, 1]]) 
    R = np.array([[1]])             
    # Initial state and covariance
    x0 = np.array([[0], [1]]) 
    P0 = np.array([[1, 0], [0, 1]]) 
    # Create Kalman Filter instance
    kf = KalmanFilter(F, B, H, Q, R, x0, P0)
    # Predict and update with the control input and measurement
    u = np.array([[1]])  
    z = np.array([[1]]) 
    # Predict step
    predicted_state = kf.predict(u)
    print("Predicted state:\n", predicted_state)
    # Update step
    updated_state = kf.update(z)
    print("Updated state:\n", updated_state)


test()

#Metodi inseriti da floppy sul codice principale
'''

def kalman_predict(x, P, A, Q):
    x = np.dot(A, x)
    P = np.dot(A, np.dot(P, A.T)) + Q
    return x, P

def kalman_update(x, P, z, H, R):
    y = z - np.dot(H, x)
    S = np.dot(H, np.dot(P, H.T)) + R
    K = np.dot(P, np.dot(H.T, np.linalg.inv(S)))
    x = x + np.dot(K, y)
    P = P - np.dot(K, np.dot(H, P))
    return x, P
'''