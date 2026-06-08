from gpiozero import Button

def init_button(pin):
    return Button(pin, pull_up=True)