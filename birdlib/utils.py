import time

def timer_start():
    return time.perf_counter()


def timer_stop(start_time):
    return time.perf_counter() - start_time