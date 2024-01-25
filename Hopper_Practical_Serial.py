# importing libraries
import serial  # Library for serial communication with python - install command: pip install pyserial
import time  # time tracking library
import serial.tools.list_ports

# teensy loop time
loop_cycle_time = 1 / 60  # 60 measurement aquistions per second; can be adjusted to max of 66Hz
control_duration = 1  # in [s] --> your anticipated flight duration / control event duration

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
        print("Not Connected")
    print("Connected successful")
    ser = serial.Serial(dev_port, baudrate=57600)
    break

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
    #!! zusätzlich .replace
    raw_data = ser.readline().decode().strip().replace('<', '').replace('>', '')
    print(raw_data)
    raw_data = raw_data.split(':')  # splitting the message

    print(raw_data)

    #!! ohne 1:
    teensy_time = int(raw_data[0])  # time in ms since teensy power on
    acceleration = float(raw_data[1]) - 9.80665  # float in [m/s^2] --> when Hopper at rest, it shows + 9.80665 m/s^2
    position = float(raw_data[2])  # float in [m]
    pressure = int(raw_data[3])  # int as 12 bit signal --> 0 corresponds to 0barg; 4095 correponds to 10barg

    alpha = 0.2  # Filter coefficient (adjust as needed)
    acceleration = (1 - alpha) * acceleration  # Filtered acceleration

    
    # print("Time: {0}".format(teensy_time))
    # print("Acc: {0}".format(acceleration))
    # print("Position: {0}".format(position))
    # print("Pressure: {0}".format(pressure))

    '''
    INSERT YOUR CONTROL LAW HERE
    
    action = control_law(.....) # action must be a 12 bit signal
    '''

    # control input which is going to be sent to teensy
    action = 0
    action = f'<1:{action}>'

    # wait until loop time has passed
    #!!!
    while time.time() - loop_timer < loop_cycle_time:
        continue

    # send control input / action to teensy
    ser.write(action.encode())

    # stop the control event once, control duration is reached
    if (time.time() - main_timer) > control_duration:
        break
#!!!
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
