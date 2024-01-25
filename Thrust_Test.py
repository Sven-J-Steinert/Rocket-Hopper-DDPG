# importing libraries
import serial # Library for serial communication with python - install command: pip install pyserial
import time  # time tracking library
import serial.tools.list_ports
import numpy as np

from TD3 import *
from datetime import datetime


# create logging file
now = datetime.now()
timestamp = now.strftime("%Y_%m_%d_-_%H_%M_%S")
filename = f"logs_{timestamp}.txt"


with open(filename, 'w') as file:
    file.write(f"counter,loop_timer,x_target,teensy_time,acceleration,position,pressure\n")

# Function to log data to a text file
def log_data(counter,loop_timer, x_target, teensy_time, acceleration, position, pressure):
    timestamp = time.time()  # Use current time as a timestamp
    with open(filename, 'a') as file:
        file.write(f"{counter},{loop_timer},{x_target},{teensy_time},{acceleration},{position},{pressure}\n")


# init agent
agent = TD3(state_dim=4, action_dim=1)
agent.load('142') # in same folder as this script
# print(agent)

# define target height
x_target = 0.5

# define step counter
counter = 0

# teensy loop time
loop_cycle_time = 0.015  # 60 measurement aquistions per second; can be adjusted to max of 66Hz
control_duration = 5  # in [s] --> your anticipated flight duration / control event duration
#%%
# ---------------------------------------------------------
# Establish Serial communication with teensy
hardware_id = '1A86:7523'  # lookup once for specific device

while True:
    connected = False
    while not connected:
        ports = serial.tools.list_ports.comports()
        for port, desc, hwid in sorted(ports):
            if hardware_id in hwid:
                dev_port = port
                connected = True
        time.sleep(0.1)
        # print("Not Connected")
    # print("Connected successful")
    ser = serial.Serial(dev_port, baudrate=57600)
    break

# ser.write(b'0')


# setup/start commands for the hopper
# resetting error status
ser.write(b'0')

# activating valve
ser.write(b'1')




# setting the reply mode --> sending the input back yes or no
# r = reply
# n = no reply
ser.write(b'n')  # don't change this

# setting the failure mode
# F = no failure
# f = failure
ser.write(b'F')  # don't change this





# -------------------------------------------------------------------------
# main loop
main_timer = time.time()  # control timer --> duration of entire control event / flight duration

while True:
    loop_timer = time.time()  # loop timer --> the time it takes for one control loop is either equivalnt to the loop_cycle_time or slower

    # signal coming from teensy
    raw_data = ser.readline().decode().strip().replace('<', '').replace('>', '')
    raw_data = raw_data.split(':')  # splitting the message

    print(raw_data)

    teensy_time = int(float(raw_data[0]))  # time in ms since teensy power on
    acceleration = float(raw_data[1]) - 9.80665  # float in [m/s^2] --> when Hopper at rest, it shows + 9.80665 m/s^2
    position = float(raw_data[2])  # float in [m]
    pressure = int(raw_data[3])  # int as 12 bit signal --> 0 corresponds to 0barg; 4095 correponds to 10barg
    pressure  = max(0,pressure)
    pressure = map(pressure, 0, 4095, 0, 10)
    # compensate 11cm offset on ground
    position = max(0, position-0.11)

    acceleration = - acceleration

    # gradually decrease the x_target to zero
    if counter >= 300:
        x_target = max(0,x_target - (0.5/300))
    state = np.array([x_target,position,acceleration,pressure]) # [x_target, x, a, p_actual]
    action = agent.select_action(state)
    action = map(action, -1, 1, 0, 4095)
    action = int(action)


    log_data(counter,loop_timer,x_target,teensy_time,acceleration,position,pressure)



    # # control input which is going to be sent to teensy
    # action = 0

    action = f'<1:{action}>'

    # wait until loop time has passed
    while time.time() - loop_timer < loop_cycle_time:
        counter += 1
        continue

    # send control input / action to teensy
    ser.write(action.encode())

    # stop the control event once, control duration is reached
    if (time.time() - main_timer) > control_duration:
        break

ser.write('<1:0>'.encode())  # shut off valve
time.sleep(0.1)
ser.write('<1:0>'.encode())  # shut off valve
time.sleep(0.1)
ser.write('<1:0>'.encode())  # shut off valve
time.sleep(0.1)
ser.write('<1:0>'.encode())  # shut off valve
time.sleep(0.1)
ser.write('<1:0>'.encode())  # shut off valve
time.sleep(0.1)
ser.write('<1:0>'.encode())  # shut off valve
time.sleep(0.1)
ser.write('<1:0>'.encode())  # shut off valve

# close serial connection
ser.close()
