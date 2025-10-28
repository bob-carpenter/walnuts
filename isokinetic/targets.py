import numpy as np



# distribution q[0] \sim N(0,1), q[1]|q[0] \sim N(q[0]^2,1)
def smileDistr(q,hessian=False):
    lp = -0.5*q[0]**2 - 0.5*(q[1]-q[0]**2)**2
    grad = np.array([-q[0] + 2.0*q[0]*q[1] - 2.0*q[0]**3,
                     q[0]**2 - q[1]])
    return [lp,grad]

def stdGauss(q,v=None):
    if(v is None):
        return -0.5*sum(q**2),-q 
    else:
        return -0.5*sum(q**2),-q,-v 
    
def funnel1(q,v=None):
    f = -(1.0/18.0)*q[0]**2 -0.5*q[1]**2*np.exp(-q[0]) - 0.5*q[0]
    g = np.array([-q[0]/9.0 + 0.5*q[1]**2*np.exp(-q[0]) - 0.5,
                  -q[1]*np.exp(-q[0])])
    if(v is None):
        return f,g
    else:
        H = np.zeros((2,2))
        H[0,0] = -1.0/9.0 - 0.5*q[1]**2*np.exp(-q[0])
        H[1,0] = q[1]*np.exp(-q[0])
        H[0,1] = H[1,0]
        H[1,1] = -np.exp(-q[0])
        return f,g,np.matmul(H,v)


def corrGauss(q,v=None):
    rho = 0.95
    f = -0.5*(q[0]**2 + q[1]**2 - 2.0*rho*q[0]*q[1])/(1.0-rho**2) 
    g = (1.0/(1.0-rho**2))*np.array([-q[0]+rho*q[1],-q[1]+rho*q[0]])
    
    if(v is None):
        return f,g
    else:
        H = np.zeros((2,2))
        H[0,0] = -1.0/(1.0-rho**2)
        H[0,1] = rho/(1.0-rho**2)
        H[1,0] = H[0,1]
        H[1,1] = H[0,0]
        return f,g,np.matmul(H,v)
    


def modFunnel(q,v=None):
    x = q[0]
    y = q[1]
    t1 = np.exp(-3.0 * x)
    t2 = 1.0 + t1
    t3 = 0.1e1 / t2
    t4 = y**2
    t5 = -0.1e1 / 0.2e1
    lp = t5 * (t2 * t4 + np.log(t3) + x**2)
    grad = np.array([0.3e1 / 0.2e1 * t1 * (t4 - t3) - x,-y * t2])
    if(v is None):
        return [lp,grad]
    else:
        H = np.zeros((2,2))
        H[0,0] = -(9.0/2.0)*(t1*t4 - 1.0/t2 + t2**(-2))  - 1.0
        H[0,1] = 3.0*t1*y
        H[1,0] = H[0,1]
        H[1,1] = -1.0 - t1
        return lp,grad,np.matmul(H,v)


