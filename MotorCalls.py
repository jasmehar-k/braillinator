# MotorCalls.py --> To handle the angles of individual motors
# Kaniesa, Hannah - Created and refined

import RPi.GPIO as GPIO
import time

# Define motor pins
servo1, servo2, servo3, servo4, servo5, servo6 = 17, 27, 22, 5, 6, 13
motor_pins = [servo1, servo2, servo3, servo4, servo5, servo6]

# Positions for motors
start_pos = [2, 24, 2, 17, 17, 9]
end_pos =   [6, 21, 5, 24, 7, 6]

# GPIO setup
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Initialize PWM instances
pwm_instances = {}

# Function to initialize PWM if not already initialized
def init_pwm():
    global pwm_instances
    for i, pin in enumerate(motor_pins):
        if pin not in pwm_instances:
            GPIO.setup(pin, GPIO.OUT)
            pwm_instances[pin] = GPIO.PWM(pin, 50)
            pwm_instances[pin].start(0)
            print(f"Initialized PWM for pin {pin}")

# Move a single servo
def move_servo(pin, angle):
    if pin not in pwm_instances:
        raise RuntimeError(f"PWM for pin {pin} is not initialized.")
    duty = angle / 18 + 2
    pwm_instances[pin].ChangeDutyCycle(duty)
    time.sleep(0.05)

# Activate motors based on pinArray
def callMotors(pinArray):
    print(f"callMotors: {pinArray}")
    init_pwm()  # Ensure PWM is initialized
    if pinArray[2] == 1:
        move_servo(motor_pins[0], end_pos[0])
    if pinArray[5] == 1:
        move_servo(motor_pins[1], end_pos[1])
    if pinArray[4] == 1:
        move_servo(motor_pins[2], end_pos[2])
    if pinArray[3] == 1:
        move_servo(motor_pins[3], end_pos[3])
    if pinArray[0] == 1:
        move_servo(motor_pins[4], end_pos[4])
    if pinArray[1] == 1:
        move_servo(motor_pins[5], end_pos[5])

# Reset motors to start positions
def resetMotors():
    print("Resetting motors...")
    for i, pin in enumerate(motor_pins):
        move_servo(pin, start_pos[i])

# Cleanup function
def cleanup():
    print("Cleaning up GPIO and stopping PWM...")
    for pin, pwm in pwm_instances.items():
        pwm.stop()
    GPIO.cleanup()


