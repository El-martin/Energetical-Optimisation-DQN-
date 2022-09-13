""" AI part of the project """
from collections import deque
from environment_HVAC import *
from scipy.optimize import least_squares, minimize


def relu(mat):
    return np.multiply(mat, (mat > 0))


def relu_derivative(mat):
    return (mat > 0) * 1


class NNLayer:
    """ Class representing a neural net layer """

    def __init__(self, input_size, output_size, activation=None, lr=1e-1):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(low=-.1, high=.1, size=(input_size, output_size))
        self.activation_function = activation
        self.lr = lr

    def forward(self, inputs, remember_for_backprop=True):
        """ Compute the output of the layer WX + B """

        # inputs has shape batch_size x layer_input_size
        input_with_bias = np.append(inputs, 1)
        unactivated = np.dot(input_with_bias, self.weights)
        # store variables for backward pass
        output = unactivated

        if self.activation_function is not None:
            # assuming here the activation function is relu
            output = self.activation_function(output)
        if remember_for_backprop:
            self.backward_store_in = input_with_bias
            self.backward_store_out = np.copy(unactivated)
        return output

    def update_weights(self, gradient):
        """ updating the weights with the gradient descent method """

        self.weights = self.weights - self.lr * gradient

    def backward(self, gradient_from_above):
        adjusted_mul = gradient_from_above
        # this is pointwise
        if self.activation_function is not None:
            adjusted_mul = np.multiply(relu_derivative(self.backward_store_out), gradient_from_above)     # f'(X)°dE/dY

        D_i = np.dot(np.transpose(np.reshape(self.backward_store_in, (1, len(self.backward_store_in)))),  # tX ° dE/dY
                     np.reshape(adjusted_mul, (1, len(adjusted_mul))))
        delta_i = np.dot(adjusted_mul, np.transpose(self.weights))[:-1]
        self.update_weights(D_i)

        return delta_i


class RLAgent:
    """ Class representing a reinforcement learning agent """

    env = None

    def __init__(self, env, in_size, out_size):
        self.env = env
        self.hidden_size = [50, 50, 50, 50, 50]
        self.input_size = in_size      # env.observation_space.shape[0]
        self.output_size = out_size    # env.action_space.n
        self.num_hidden_layers = 5
        self.epsilon = 1.
        self.memory = deque([], 1000000)
        self.gamma = 0.99

        self.observation = None

        self.layers = [NNLayer(self.input_size + 1, self.hidden_size[0], activation=relu)]
        for i in range(self.num_hidden_layers - 1):
            self.layers.append(NNLayer(self.hidden_size[i] + 1, self.hidden_size[i+1], activation=relu))
        self.layers.append(NNLayer(self.hidden_size[-1] + 1, self.output_size))

    @staticmethod
    def normalize(observation):
        """ Normalizing the data before input """

        o = observation
        weights = [1/10, 1/10, 1/10, 1/10, 1, 1, 1, 1, 1]
        return np.array([o[i] * weights[i] for i in range(9)])

    def net(self, action):
        """ auxiliary to self.opt_vect """
        input = self.normalize(np.concatenate((self.observation, action)))
        return -self.forward(input)

    def opt_vect(self, observation):
        """ Optimal action according to the network """

        self.observation = observation
        result = least_squares(self.net, np.array([0.01, 0.01, 0.01, 0.01, 1]),
                               bounds=([0, 0, 0, 0, 0], [1, 1, 1, 1, 2]), xtol=1e-6)

        if not result.success:
            print("optimization failed")
            return np.array([0., 0., .5, .5, 1])

        return result.x

    def max_rew(self, observation):
        """ Maximum expectable reward according to the network """

        self.observation = observation
        return minimize(self.net, np.array([0, 0, 0, 0, 1]), bounds=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 2)]).fun

    def select_action(self, observation, best=False):
        """ Select an action, randomly or not """

        if np.random.random() > self.epsilon or best:
            self.observation = observation
            opt_vector = self.opt_vect(observation)
            # print(f"optimal action {opt_vector}")
            return opt_vector
        else:
            return np.concatenate((np.random.rand(4), 2*np.random.rand(1)))

    def forward(self, observation, remember_for_backprop=True):
        """ Compute the network's estimation of the (state, action) value"""

        vals = np.copy(observation)
        for layer in self.layers:
            vals = layer.forward(vals, remember_for_backprop)
        return vals

    def remember(self, done, action, reward, observation, prev_obs):
        self.memory.append([done, action, reward, observation, prev_obs])

    def experience_replay(self, update_size=20, show_delta=False):
        """ Learning from past experiences randomly sampled from the agent's memory,
        adapted to continuous action spaces """

        if len(self.memory) < update_size:
            return
        else:
            batch_indices = np.random.choice(len(self.memory), update_size)
            for index in batch_indices:
                done, action_selected, new_obs, reward, prev_obs = self.memory[index]

                input = self.normalize(np.concatenate((prev_obs, action_selected)))

                action_value = self.forward(input, remember_for_backprop=True)

                if done:
                    experimental_value = -1
                else:
                    experimental_value = reward + self.gamma * self.max_rew(new_obs)
                self.backward(action_value, experimental_value, show_delta)

        self.epsilon = self.epsilon if self.epsilon < 0.005 else self.epsilon * .99
        for layer in self.layers:
            layer.lr = layer.lr if layer.lr < 1e-5 else layer.lr * 0.99

    def backward(self, calculated_values, experimental_values, show_delta):
        """ backpropagation """

        delta = (calculated_values - experimental_values)
        if show_delta:
            print('delta = {}'.format(delta))
        for layer in reversed(self.layers):
            delta = layer.backward(delta)
