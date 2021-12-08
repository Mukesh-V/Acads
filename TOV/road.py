import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

'''
    wheelbase     Distance between front and rear axles                 m
    speed         Speed of the car                                      km/h
    amplitude     Amplitude of bump/pothole                             mm
    type          0-halfsine, 1-square_trough, 2-triangle(maybe?)
'''
def road_profile(options):
    options['speed'] *= 5/18
    bump_length = 1  # in meters
    time_taken = bump_length/options['speed']
    travel_time = options['wheelbase']/options['speed']
    omega = np.pi / time_taken

    def f(t):
        '''
            x1, x2 = Front Left and Right
            x3, x4 = Back Left and Right
        '''
        x1 = x2 = x3 = x4 = 0.0
        if t > 0 and t <= time_taken:
            if options['type'] == 0:
                x1 = options['amplitude'] * -1
            elif options['type'] == 1:
                x1 = options['amplitude'] * np.sin(omega*t)

        elif t > travel_time and t <= travel_time + time_taken:
            if options['type'] == 0:
                x3 = options['amplitude'] * -1
            elif options['type'] == 1:
                x3 = options['amplitude'] * np.sin(omega*(t-travel_time))

        if options['side'] == 2:
            x2 = x1
            x4 = x3

        return np.array([x1, x2, x3, x4])

    return f

if __name__ == "__main__":
    sqr_options = {
        'wheelbase': 2,
        'speed': 20,
        'amplitude': 0.1,
        'type': 0,
        'side': 1
    }
    sine_options = sqr_options.copy()
    sine_options['type'] = 1

    sqr_road = road_profile(sqr_options)
    sine_road = road_profile(sine_options)

    U_sqr, U_sine = [], []
    T = np.arange(0, 1, 0.01)
    for i in T:
        U_sqr.append(sqr_road(i))
        U_sine.append(sine_road(i))

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    ax1.plot(T, U_sqr)
    ax1.set_ylabel('Tyre displacement (in meters)')
    ax1.set_xlabel('Time (in seconds)')
    ax1.legend(["Left Front", "Right Front", "Left Back", "Right Back"], loc ="lower right")

    ax2.plot(T, U_sine)
    ax2.set_xlabel('Time (in seconds)')
    ax2.legend(["Left Front", "Right Front", "Left Back", "Right Back"], loc ="lower right")

    plt.show()