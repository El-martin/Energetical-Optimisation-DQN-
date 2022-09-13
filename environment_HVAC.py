import numpy as np


class HVACunit:
    """ Device controlling the temperature (T) and humidity (h) parameters inside the building. """

    def __init__(self, T_hot, T_cold, elec_price):
        self.T_hot = T_hot
        self.T_cold = T_cold
        self.elec_price = elec_price
        self.valves = {"h_water": 0., "c_water": 0., "in_air": 0., "out_air": 0., "h_control": 0.}

        self.cost_activation = lambda x: 1 / (1 + np.exp(- 1.25 * x + 3.5))

    def mix(self, weather):           # weather = [T_in, T_out, h_in, h_out]

        T_air, h, T_liq = weather[0], weather[2]*self.valves["h_control"], None

        # air in & out mix
        n_open_air = 2 - [self.valves["in_air"], self.valves["out_air"]].count(0)
        if n_open_air != 0:
            D_air = (self.valves["in_air"] + self.valves["out_air"]) / n_open_air
        else:
            D_air = 0

        if D_air != 0:
            T_air = (weather[0] * self.valves["in_air"] + weather[1] * self.valves["out_air"])/(n_open_air * D_air)

        # water system mix
        n_open_liq = 2 - [self.valves["h_water"], self.valves["c_water"]].count(0)
        if n_open_liq != 0:
            D_liq = (self.valves["h_water"] + self.valves["c_water"]) / n_open_liq
        else:
            D_liq = 0

        if D_liq != 0:
            T_liq = (self.T_hot * self.valves["h_water"] + self.T_cold * self.valves["c_water"]) / (n_open_liq * D_liq)

        # humidity mix
        n_open_hum = 2 - [self.valves["in_air"], self.valves["out_air"]].count(0)  # , self.valves["steam"]
        if n_open_hum != 0:
            D_hum = (self.valves["in_air"] + self.valves["out_air"]) / n_open_hum  # + self.valves["steam"]
        else:
            D_hum = 0

        if D_hum != 0:
            h = (weather[2] * self.valves["in_air"] + weather[3] * self.valves["out_air"]) / \
                (D_hum * n_open_hum) * self.valves["h_control"]

        if D_liq == 0:
            T = T_air
        else:
            T = .9*T_liq + .1*T_air if D_air != 0 else T_liq   # Huge modification ( T = T_liq)

        return T, h

    def modify_valves(self, openings):
        keys = list(self.valves.keys())
        for i in range(len(keys)):
            self.valves[keys[i]] = openings[i]

    def elec_cost(self):
        weights = [1.2, 1, 1.1, 1.1]   # humidification is neglected
        return self.cost_activation(sum([list(self.valves.values())[i] * weights[i] for i in range(4)]))


class Environment:
    """ Consists of a building exchanging heat with the outside
    parameterized by the weather, and a HVAC unit. """

    def __init__(self, T_in, T_out, h_in, weather_data, hT_unit, T_lim, h_lim, rew_policy):
        self.T_in = T_in
        self.T_out = T_out
        self.h_in = h_in
        self.weather_data = weather_data
        self.date = 0

        self.hT_unit = hT_unit

        self.rew_policy = rew_policy      # [Temp weight, hum weight, accuracy weight, expense weight]

        self.T_lim = T_lim
        self.h_lim = h_lim

        T_min, T_max = T_lim[0], T_lim[1]
        h_min, h_max = h_lim[0], h_lim[1]

        nu_T = (T_max - T_min) / 2
        T_moy = (T_max + T_min) / 2
        nu_h = (h_max - h_min) / 2
        h_moy = (h_max + h_min) / 2

        self.g1 = lambda T: 2 * np.exp(np.log(1/2) * (T - T_moy)**2 / nu_T**2) - 1
        self.g2 = lambda h: 2 * np.exp(np.log(1/2) * (h - h_moy)**2 / nu_h**2) - 1

    def reset(self, T_in):
        self.date = 0
        init_weather = self.weather_data[0]
        self.T_in = T_in
        self.T_out = init_weather[3]

        return np.array([T_in, self.T_out, self.h_in, init_weather[2]])

    def is_over(self, T, h):
        return not (self.T_lim[0] <= T <= self.T_lim[1] and self.h_lim[0] <= h <= self.h_lim[1])

    def reward(self, T, h):
        w1, w2, w3, w4 = self.rew_policy[0], self.rew_policy[2], self.rew_policy[2], self.rew_policy[3]

        p = w1 * self.g1(T) + w2 * self.g2(h)   # w1 + w2 = 1
        e = self.hT_unit.elec_cost()

        r = w3 * p - w4 * e                     # w3 + w4 = 1

        return r

    def step(self, at, dt=6):
        """ Computing the next physic state and returning a reward to the agent """
        # st = [T_in, T_out, h_in, h_out]   at = [hot water, c water, air in, air out, humidity]

        mc = 1.3e3  # J/K
        tau = 12    # min
        R = 1.1e-2  # K/W

        i = self.date
        st = [self.T_in, self.weather_data[i][3], self.h_in, self.weather_data[i][2]]

        self.hT_unit.modify_valves(at)
        T_cond, h_cond = self.hT_unit.mix(st)

        # New Temperature
        phi_prod = mc/(tau*60) * (T_cond - st[0])
        T_loss = (st[1] - st[0]) * (1 - np.exp(-dt*60/(R*mc)))   # dt in minutes, see formula

        T_new = st[0] + phi_prod*dt*60 / mc + T_loss

        # New Humidity
        h_new = (h_cond - st[2]) * dt / tau + st[2]

        self.T_in, self.h_in = T_new, h_new

        # Did the agent lose ?
        done = self.is_over(T_new, h_new)
        if done:
            rew = -1
        else:
            rew = self.reward(T_new, h_new)

        # next data point
        self.date += 1
        curr_weather = self.weather_data[self.date]
        stp1 = np.array([T_new, curr_weather[3], h_new, curr_weather[2]])

        return rew, stp1, done
