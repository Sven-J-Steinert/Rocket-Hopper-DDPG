from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import CoolProp.CoolProp as CP

# some global variables
m = 3.5+1 # kg
F_RR = 10 # N
k  = 6 # N /m
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
        self.observation_space = Box(low=np.array([0.,5., -100.]), high=np.array([0.,5., 100.]), shape=(3,)) # [x_target,x,a]
        # Set simulation time
        self.sim_time = 16 # [s]
        self.p_set_old = 0 # [bar]
        self.v_old = 0
        self.check = False

        # test trajectory: select 3 random hover points and landing in the end
        self.x_target = random.uniform(1, 3)

        
        # Set start altitude and velocity
        self.y = np.array([0.,0.] , dtype=np.float32)
        self.state = np.array([self.x_target,0.,0.] , dtype=np.float32)
        self.error_old = self.x_target - self.y[0]

        
    def step(self, action):
        
        #print('time',self.sim_time,'x_target',self.x_target)
        #print('x_target',self.x_target)
        
        
        # Apply action

        # action is the pressure
        p_set = action
        p_actual = dynamic_restriction(p_set,self.p_set_old)

        y = self.sim_step(self.y,p_actual)
        self.y = y[:-1]

        self.state = np.array([self.x_target,y[0],y[2]]) # craft [x_target,x,a]
        
        # Reduce sim_time by a step
        self.sim_time -= 1/60

        # reward small error in gauss dist
        sigma = 0.5 # variance - gauss parameter
        reward = 1/(sigma*np.sqrt(2*np.pi))* np.exp(-0.5*((self.state[1]-self.x_target)/(sigma))**2)
        
        # penalize fast change in p_set
        #if abs(self.p_set_old - p_set) > 0.5:
        #    reward += -10 

        # update for next loop
        self.p_set_old = p_actual
        
        # Check if shower is done
        if self.sim_time <= 0: 
            done = True
        else:
            done = False
        
        # Apply temperature noise
        #self.state += random.randint(-1,1)
        # Set placeholder for info
        info = [p_actual,self.y[1]]
        
        # Return step information
        return self.state, reward, done, info


    def render(self):
        # Implement viz
        pass
    
    def reset(self):
        # Reset all variables
        self.state = np.array([self.x_target,0,0])
        self.y = np.array([0,0])
        self.p_set_old = 0
        self.v_old = 0
        self.error_old = self.x_target - self.y[0]
        self.x_target = random.uniform(1, 3)
        self.sim_time = 16 # [s]
        self.check = False
        return self.state
    
    def sim_step(self,y,p):
        p = np.squeeze(p) # make sure its one dimensional
        h = 1/600              # stepsize in seconds
        t0 = 0                  # initial time in seconds
        tn = 1/60               # final time in seconds
        
        time = np.linspace(t0, tn, int((tn-t0)/h)+1)
    
        # update Thrust
        global F_T
    
        F_T = F_Thrust_fast(p)
        
        for t in time:
            y = rk4_e(ode, y, h, t)
            
        x = y[0]
        v = y[1]
        a = v-self.v_old
    
        self.v_old = v
            
        return np.array([x,v,a])

# resistance modelling - smoothed to stablize model
def F_R(v):
    v_thr = 0.01
    
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


def dynamic_restriction(p_set,p_set_old):
    # make both inputs scalar
    p_set = np.squeeze(p_set)
    p_set_old = np.squeeze(p_set_old)
    
    # main valve opening / closing restriction
    # 0 % = 0 bar to 100% = 10 bar
    
    dp = (p_set-p_set_old) # [bar]
    sim_step = 1/60 # [s]
    dp_dt = dp/sim_step # [bar/s]
    
    direction = (dp > 0) # True = opening, False = closing

    # opening restrictions
    # dp/dt = 10 bar / 0.825 s
    dp_dt_opening = 10 / 0.825

    # closing restrictions
    # dp/dt = 10 bar / 1.7 s
    dp_dt_closing = 10 / 1.7

    # if requested change in pressure is to high
    # return maximum allowed pressure over the next sim step (1/60 s)

    if direction:
        # opening
        if dp_dt > dp_dt_opening:
            # compute max allowed for new p_set
            p_set = dp_dt_opening * sim_step + p_set_old
            
    if not direction:
        # closing
        if dp_dt > dp_dt_closing:
            # compute max allowed for new p_set
            p_set = dp_dt_closing * sim_step + p_set_old

    return p_set
