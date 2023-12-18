from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import CoolProp.CoolProp as CP

# some global variables
g0 = 9.80665

F_T = 0 # define global var

def F_Thrust_fast(p_valve):
    coefficients = np.array([10.600789308865751,-9.503317777109121])
    linear_fit = np.poly1d(coefficients)
    return max(0,linear_fit(p_valve))

class HopperEnv(Env):
    def __init__(self):
        # Actions we can take: set pressure between 0 and 7 bar
        self.action_space = Box(low=0.0, high=7.0, shape=(1,), dtype=np.float32) # pressure
        # Altitude range
        self.observation_space = Box(low=np.array([0.,5., -10.]), high=np.array([0.,5., 10.]), shape=(3,)) # [x_target,x,a]
        # Set simulation time
        self.sim_time = 5 # [s] total sim time
        self.tn = 1/60    # [s] step sim time
        self.h = 1/120    # [s] stepsize ode
        
        self.p_actual = 0 # [bar]
        self.counter = 0 # for constant penalty
        self.x_old = 0 # for constant penalty
        self.v_old = 0

        # test trajectory: select 3 random hover points and landing in the end
        self.x_target = random.uniform(1, 3)
        self.x_target = 2

        # Set start altitude and velocity
        self.y = np.array([0.,0.])
        self.state = np.array([self.x_target,0.,0.])


    def reset(self):
        # Reset all variables
        self.state = np.array([self.x_target,0,0])
        self.y = np.array([0.,0.])
        self.p_actual = 0
        self.v_old = 0
        self.x_target = random.uniform(1, 3)
        self.x_target = 2
        self.sim_time = 5 # [s]

        return self.state

        
    def step(self, action):
        
        #print('time',self.sim_time,'x_target',self.x_target)
        #print('x_target',self.x_target)
        
        
        # Apply action
        p_set = action
        self.p_actual = valve_behavior(p_set,self.p_actual)

        y = self.sim_step(self.y,self.p_actual)
        self.y = y[:-1]

        self.state = np.array([self.x_target,y[0],y[2]]) # craft [x_target,x,a]
        
        # Reduce sim_time by a step
        self.sim_time -= self.tn

        # reward small error in gauss dist

        error = self.state[1]-self.x_target
        scale = 1
        sigma = 0.5 # variance - gauss parameter
        reward = scale*(1/(sigma*np.sqrt(2*np.pi))* np.exp(-0.5*((error)/(sigma))**2)) # small error gauss

        scale = 10
        sigma = 0.5 # variance - gauss parameter
        
        #reward = reward * (1 + scale*(1/(sigma*np.sqrt(2*np.pi))* np.exp(-0.5*((y[2])/(sigma))**2))) # small velocity gauss
        
        
        #if p_set == self.p_set_old and abs(y[0] - self.x_old) > 0.001:
        #    reward -= self.counter * 0.005
        #    self.counter += 1
        #else:
        #    self.counter = 0

        self.x_old = y[0]
        
        # penalize fast change in p_set
        #if abs(self.p_set_old - p_set) > 0.5:
        #    reward += -10 

        # update for next loop

        
        # Check if shower is done
        if self.sim_time <= 0: 
            done = True
        else:
            done = False
        
        # Apply sensor noise
        
        #self.state += random.randint(-1,1)
        # Set placeholder for info
        info = [self.p_actual,self.y[1]]
        
        # Return step information
        return self.state, reward, done, info


    def render(self):
        # Implement viz
        pass
    
    def sim_step(self,y,p):
        p = np.squeeze(p) # make sure its one dimensional
        
        t0 = 0                  # initial time in seconds
        
        time = np.linspace(t0, self.tn, int((self.tn-t0)/self.h)+1)
    
        # update Thrust
        global F_T
    
        F_T = F_Thrust_fast(p)
        
        for t in time[1:]:
            y = rk4_e(ode, y, self.h, t)
            
        x = y[0]
        v = y[1]
        a = v-self.v_old
    
        self.v_old = v
            
        return np.array([x,v,a])

# resistance modelling - smoothed to stablize model
def F_R(v):
    v_thr = 0.01
    F_RR = 10 # N
    
    if abs(v) > v_thr:
        F_R = np.sign(v)*F_RR
    else:
        F_R = (abs(v)/v_thr)*np.sign(v)*F_RR
    return F_R


def ode(t, y):
    """
    Defines the system of ODEs for free fall.

    Parameters:
        t (float): Time.
        y (array): Array containing the position and velocity [x, v].

    Returns:
        dydt (array): Array containing the derivatives [v, a].
    """
    global F_T
    
    x = y[0]
    v = y[1]

    k  = 6 # N /m
    m = 3.5+1 # kg
    
    a = (1/m)*(F_T - (m*g0) - (k *x) - (F_R(v)))
    # restrict movement to be not able to go below 0 in position
    if (x == 0 or x < 0 ) and a < 0:
        # would move into negative realm
        a = 0
    
    dydt = np.array([v, a])
    return dydt

def rk4_e(f, y, h, t):
    # runge kutta 4th order explicit
    tk_05 = t + 0.5*h
    yk_025 = y + 0.5 * h * f(t, y)
    yk_05 = y + 0.5 * h * f(tk_05, yk_025)
    yk_075 = y + h * f(tk_05, yk_05)
    
    return y + h/6 * (f(t, y) + 2 * f(tk_05, yk_025) + 2 * f(tk_05, yk_05) + f(t+h, yk_075))


def valve_behavior(p_set,p_current):
    # make both inputs scalar
    p_set = np.squeeze(p_set)
    p_current = np.squeeze(p_current)

    freq=1/60 #[s^-1]
    
    T=0.2

    # explicit computation of value after freq time
    p_actual = p_set*(1-np.exp(-freq/T))+p_current*np.exp(-freq/T)
    
    return p_actual
