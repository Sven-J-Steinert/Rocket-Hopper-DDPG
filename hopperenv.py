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

class HopperEnv(Env):
    def __init__(self):
        # Actions we can take: set pressure between 0 and 7 bar
        self.action_space = Box(low=0.0, high=7.0, shape=(1,), dtype=np.float32)
        # Altitude range
        self.observation_space = Box(low=np.array([0., -100.]), high=np.array([5., 100.]), shape=(2,))
        # Set start altitude and velocity
        self.state = np.array([0.,0.] , dtype=np.float32)
        # Set simulation time
        self.sim_time = 40 # [s]
        self.p_set_old = 0 # [bar]
        self.x_target = 2. # [m]
        
    def step(self, action):
        # Apply action

        # action is the pressure
        p_set = action
        #p_actual = dynamic_restriction(p_set,self.p_set_old)
        p_actual = p_set
        self.state = sim_step(self.state,p_actual)
        self.p_set_old = p_set
        
        # Reduce sim_time by a step
        self.sim_time -= 1/60
        
        # Calculate reward
        reward = 0
        error = abs(self.state[0] - self.x_target)
        threshold = 2.0
        if error < threshold: 
            if error == 0:
                reward += 1000
            else:
                reward += min(1000,threshold/error)
        else: 
            reward += -1 
        
        # Check if shower is done
        if self.sim_time <= 0: 
            done = True
        else:
            done = False
        
        # Apply temperature noise
        #self.state += random.randint(-1,1)
        # Set placeholder for info
        info = {}
        
        # Return step information
        return self.state, reward, done, info

    def render(self):
        # Implement viz
        pass
    
    def reset(self):
        # Reset shower temperature
        self.state = np.array([0,0])
        # Reset shower time
        self.sim_time = 40 # [s]
        return self.state
    
    
    
def sim_step(y,p=None):
    h = 1/600              # stepsize in seconds
    t0 = 0                  # initial time in seconds
    tn = 1/60               # final time in seconds
    
    time = np.linspace(t0, tn, int((tn-t0)/h)+1)

    # update Thrust
    global F_T
    F_T = F_Thrust_fast(p)
    
    for t in time:
        y = rk4_e(ode, y, h, t)
        
    return y

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

def F_Thrust_fast(p_valve):
    coefficients = [10.60078931 -9.50331778]
    linear_fit = np.poly1d(coefficients)
    return max(0,linear_fit(p_valve))

def F_Thrust_NASA(p_valve):
    R = 296.8 # Gas constant of Nitrogen
    gamma = 1.4
    D_th = 0.010 # nozzle throat diameter m
    D_ex = 0.011 # nozzle exit diameter mm
    A_th = (np.pi/4) * D_th**2 # [m²]
    A_ex = (np.pi/4) * D_ex**2 # [m²]
    
    H = 263289.1858405551 # H = CP.PropsSI('H','P',300 * 1e5,'T',293, "Nitrogen")

    # values over the mach shock
    p_1 = p_valve * 1e5 # [Pa]
    T_1 = CP.PropsSI('T','P',p_1,'H',H, "Nitrogen") # [K]

    md =  (A_th * p_1/np.sqrt(T_1)) * np.sqrt(gamma/R) * ((gamma + 1)/2)**-((gamma + 1)/(gamma - 1)/2) 

    # converge Mach_exit
    M_ex = 2 # Initial value for the exit Mach number
    error = np.inf
    Aex_Ath_target = A_ex / A_th

    M_ex = 1.84790
    
    T_2 = T_1 * ((1+ ((gamma-1)/2)*M_ex**2)**(-1))
    p_2 = p_1 * ((1+ ((gamma-1)/2)*M_ex**2)**(-((gamma)/(gamma-1))))

    v_ex = M_ex * np.sqrt(gamma*R*T_2)

    p_infinity = 1 * 1e5 # [Pa]

    F_T = md * v_ex + (p_2 - p_infinity) * A_ex

    return F_T