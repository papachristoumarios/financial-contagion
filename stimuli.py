import numpy as np

def stimulus_check(wealth, num_household):

    if num_household == 1:
        if wealth <= 75000:
            return 1200
        elif 75000 <= wealth <= 150000:
            return 1200 * (150000 - wealth) / (75000)
        else:
            return 0
    elif num_household >= 2:
        stimulus_unscaled = 2400 + 500 * (num_household - 2)
        if wealth <= 150000:
            return stimulus_unscaled
        elif 150000 <= wealth <= 300000:
            return stimulus_unscaled * (300000 - wealth) / 150000
        else:
            return 0 

    return 0

