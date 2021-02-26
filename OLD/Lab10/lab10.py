import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rnd
import math
from scipy import special

# global variables
c = 3 * (10 ** 8)                # Speed of light
c_network = 2/3 * c              # Speed of signal on the network

json_path1 = 'Resources/nodes_full_fixed_rate.json'  # Json file address
json_path2 = 'Resources/nodes_full_flex_rate.json'  # Json file address
json_path3 = 'Resources/nodes_full_shannon.json'  # Json file address

snr_or_latency_choice = 'snr'

number_of_connections = 200

input_signal_power = 0.001

number_of_channels = 10
gain = 16  # dB
noise_figure = 3  # dB
gain_linear = 10 ** (gain / 10)  # Linear
noise_figure_linear = 10 ** (noise_figure / 10)  # Linear
h = 6.62607015e-34  # Plack constant  J * s
f = 193.414e12  # C-band center frequency
alpha = 0.2  # dB/km
beta2 = 2.13e-26  # (m Hz^2)^-1
gamma = 1.27e-3  # (m W)^-1
# f_max = 191.2e12
# f_min = 195.6e12
# df = (f_max - f_min) / number_of_channels
df = 50e9
Rs = 32e9
Bn = 12.5e9
BERt = 10e-3


# Models the lightpath physical parameters: contains signal power, signal path, noise power,
# lightpath latency and the selected channel
class Lightpath:
    def __init__(self, power, path, channel):
        self._signal_power = power
        self._path = path
        self._noise_power = 0.0
        self._latency = 0.0
        self._channel = channel
        self._starting_node = path[0]
        self._previous_node = path[0]
        self._df = df
        self._Rs = Rs

    @property
    def signal_power(self):
        return self._signal_power

    @signal_power.setter
    def signal_power(self, signal_power):
        self._signal_power = signal_power

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        self._path = path

    @property
    def noise_power(self):
        return self._noise_power

    @noise_power.setter
    def noise_power(self, noise_power):
        self._noise_power = noise_power

    @property
    def latency(self):
        return self._latency

    @latency.setter
    def latency(self, latency):
        self._latency = latency

    @property
    def channel(self):
        return self._channel

    @property
    def starting_node(self):
        return self._starting_node

    @property
    def previous_node(self):
        return self._previous_node

    @previous_node.setter
    def previous_node(self, previous_node):
        self._previous_node = previous_node

    @property
    def df(self):
        return self._df

    @property
    def Rs(self):
        return self._Rs

    # Updates noise power
    def add_noise(self, noise):
        self.noise_power += noise

    # Updates lightpath latency
    def add_latency(self, latency):
        self.latency += latency

    # Updates signal path
    def next(self):
        self.path = self.path[1:]


# Models the nodes of the network: contains the name of the node, its position, the list of the connected nodes and
# lines and the switching matrix
class Node:
    def __init__(self, node_dict):
        self._label = node_dict['label']
        self._position = node_dict['position']
        self._connected_nodes = node_dict['connected_nodes']
        self._successive = {}
        self._switching_matrix = {}
        self._transceiver = 'fixed-rate'

    @property
    def label(self):
        return self._label

    @property
    def position(self):
        return self._position

    @property
    def connected_nodes(self):
        return self._connected_nodes

    @property
    def successive(self):
        return self._successive

    @successive.setter
    def successive(self, successive):
        self._successive = successive

    @property
    def switching_matrix(self):
        return self._switching_matrix

    @switching_matrix.setter
    def switching_matrix(self, switching_matrix):
        self._switching_matrix = switching_matrix

    @property
    def transceiver(self):
        return self._transceiver

    @transceiver.setter
    def transceiver(self, transceiver):
        self._transceiver = transceiver

    # Propagates the signal information along the node
    def propagate(self, signal_information):
        path = signal_information.path
        if len(path) > 1:
            if signal_information.starting_node != path[0]:
                if signal_information.channel == number_of_channels-1:
                    self.switching_matrix[signal_information.previous_node][path[1]][signal_information.channel-1] = 0
                elif signal_information.channel == 0:
                    self.switching_matrix[signal_information.previous_node][path[1]][signal_information.channel + 1] = 0
                else:
                    self.switching_matrix[signal_information.previous_node][path[1]][signal_information.channel + 1] = 0
                    self.switching_matrix[signal_information.previous_node][path[1]][signal_information.channel - 1] = 0

            line_label = path[:2]
            line = self.successive[line_label]
            signal_information.previous_node = path[0]
            signal_information.next()
            signal_information.signal_power = line.optimized_launch_power()
            signal_information = line.propagate(signal_information)
        return signal_information


# Models the lines of the network: contains the name of the line, its length, the list of the connected nodes and
# state of the channels of the line
class Line:
    def __init__(self, line_dict):
        self._label = line_dict['label']
        self._length = line_dict['length']
        self._successive = {}
        self._state = list()
        self._n_amplifiers = line_dict['n_amplifiers']
        self._gain = gain
        self._gain_linear = 10 ** (self._gain / 10)  # Linear
        self._noise_figure = noise_figure
        self._noise_figure_linear = 10 ** (self._noise_figure / 10)  # Linear
        self._alpha = line_dict['alpha']
        self._alpha_linear = (self._alpha / (20 * math.log10(math.e))) / 1000
        self._beta2 = beta2
        self._gamma = gamma

        for i in range(number_of_channels):
            self._state.append(1)

    @property
    def label(self):
        return self._label

    @property
    def length(self):
        return self._length

    @property
    def successive(self):
        return self._successive

    @successive.setter
    def successive(self, successive):
        self._successive = successive

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        self._state = state

    @property
    def n_amplifiers(self):
        return self._n_amplifiers

    @n_amplifiers.setter
    def n_amplifiers(self, n_amplifiers):
        self._n_amplifiers = n_amplifiers

    @property
    def gain(self):
        return self._gain

    @property
    def gain_linear(self):
        return self._gain_linear

    @property
    def noise_figure(self):
        return self._noise_figure

    @property
    def noise_figure_linear(self):
        return self._noise_figure_linear

    @property
    def alpha(self):
        return self._alpha

    @property
    def alpha_linear(self):
        return self._alpha_linear

    @property
    def beta2(self):
        return self._beta2

    @property
    def gamma(self):
        return self._gamma

    # Generates the latency of the line
    def latency_generation(self):
        latency = self.length / c_network
        return latency

    # Generates the noise of the line
    def noise_generation(self, signal_power):
        noise = self.nli_generation(signal_power) + self.ase_generation()
        return noise

    # Generates the ase noise of the line
    def ase_generation(self):
        ase = self.n_amplifiers * h * f * Bn * self.noise_figure_linear * (self.gain_linear - 1)
        return ase

    # Generates the non linear noise of the line
    def nli_generation(self, signal_power):
        eta_nli = 16 / (27 * math.pi) * \
                  math.log(((math.pi ** 2) / 2) * ((self.beta2 * (Rs ** 2)) / self.alpha_linear) *
                           (number_of_channels ** ((2 * Rs) / df)), math.e) * \
                  ((self.gamma ** 2) / (4 * self._alpha_linear * self.beta2 * (Rs ** 3)))
        n_span = self.n_amplifiers - 1
        nli = (signal_power ** 3) * eta_nli * n_span * Bn
        return nli

    # Calculates the optimal launch power
    def optimized_launch_power(self):
        eta_nli = 16 / (27 * math.pi) * \
                  math.log(((math.pi ** 2) / 2) * ((self.beta2 * (Rs ** 2)) / self.alpha_linear) *
                           (number_of_channels ** ((2 * Rs) / df)), math.e) * \
                  ((self.gamma ** 2) / (4 * self._alpha_linear * self.beta2 * (Rs ** 3)))
        p_opt = (self.ase_generation() / (2 * eta_nli * Bn)) ** (1. / 3.)
        return p_opt

    # Propagates the signal information along the line
    def propagate(self, signal_information):
        # Update latency
        latency = self.latency_generation()
        signal_information.add_latency(latency)

        # Update noise
        signal_power = signal_information.signal_power
        noise = self.noise_generation(signal_power)
        signal_information.add_noise(noise)

        # Update line occupancy
        self.state[signal_information.channel] = 0

        node = self.successive[signal_information.path[0]]
        signal_information = node.propagate(signal_information)
        return signal_information


# Models the the network: contains the route space, the list of all the paths with the related snr/latency
# (in weighted paths), the nodes and the lines of the network and the list of nodes and lines belonging to
# every possible path
class Network:
    def __init__(self, json_path):
        self._route_space = pd.DataFrame()
        self._weighted_paths = pd.DataFrame()
        node_json = json.load(open(json_path, 'r'))
        self._nodes = {}
        self._lines = {}
        self._line_list_dict = {}
        self._node_list_dict = {}
        self._path_list = list()
        self._switching_matrix_dict = {}

        # Creation of node and line instances
        for node_label in node_json:
            # Create the node instance
            node_dict = node_json[node_label]
            node_dict['label'] = node_label
            node = Node(node_dict)
            self._nodes[node_label] = node
            self.switching_matrix_dict[node_label] = node_dict['switching_matrix']
            for connected_node1_label in self.switching_matrix_dict[node_label]:
                for connected_node2_label in self.switching_matrix_dict[node_label][connected_node1_label]:
                    self.switching_matrix_dict[node_label][connected_node1_label][connected_node2_label] = \
                        np.array(self.switching_matrix_dict[node_label][connected_node1_label][connected_node2_label])
            self.nodes[node_label].transceiver = node_json[node_label]['transceiver']

            # Create the line instances
            for connected_node_label in node_dict['connected_nodes']:
                line_dict = {}
                line_label = node_label + connected_node_label
                line_dict['label'] = line_label
                node_position = np.array(node_json[node_label]['position'])
                connected_node_position = np.array(node_json[connected_node_label]['position'])
                line_dict['length'] = np.sqrt(np.sum((node_position-connected_node_position)**2))
                line_dict['n_amplifiers'] = math.ceil(line_dict['length'] / 80000) + 1  # booster, pre-amplifier and in-line amplifiers
                line_dict['alpha'] = alpha
                line = Line(line_dict)
                self._lines[line_label] = line

    @property
    def weighted_paths(self):
        return self._weighted_paths

    @weighted_paths.setter
    def weighted_paths(self, df):
        self._weighted_paths = df

    @property
    def nodes(self):
        return self._nodes

    @property
    def lines(self):
        return self._lines

    @property
    def route_space(self):
        return self._route_space

    @route_space.setter
    def route_space(self, route_space):
        self._route_space = route_space

    @property
    def line_list_dict(self):
        return self._line_list_dict

    @property
    def node_list_dict(self):
        return self._node_list_dict

    @property
    def path_list(self):
        return self._path_list

    @property
    def switching_matrix_dict(self):
        return self._switching_matrix_dict

    # Initializes the network class
    def initialize(self, json_path):
        # Creates a data frame that will be filled with all the possible paths information.
        df = pd.DataFrame()
        node_labels = self._nodes.keys()
        pairs = []
        for label1 in node_labels:
            for label2 in node_labels:
                if label1 != label2:
                    pairs.append(label1 + label2)

        # Columns = ['path', 'latency', 'noise', 'snr']
        paths = []
        latencies = []
        noises = []
        snrs = []
        for pair in pairs:
            for path in self.find_paths(pair[0], pair[1]):
                # Generation of the path strings
                self.path_list.append(path)
                path_string = ''
                for node in path:
                    path_string += node + '->'
                paths.append(path_string[:-2])

                # Propagation of the signal along all the possible paths
                # Gets latency, snr and noise power of all the possible paths
                signal_information = Lightpath(input_signal_power, path, 0)
                signal_information = self.propagate(signal_information)
                latencies.append(signal_information.latency)
                noises.append(signal_information.noise_power)
                snrs.append(
                    10 * np.log10(
                        signal_information.signal_power / signal_information.noise_power
                    )
                )

        df['path'] = paths
        df['latency'] = latencies
        df['noise'] = noises
        df['snr'] = snrs
        # Saves the contend of the data frame in weighted paths
        self.weighted_paths = df

        # Generates for all the possible paths a list of all the nodes
        # and a list of all the lines belonging to that path
        self.generate_node_and_line_list()

        # Cleans switching matrix and line state
        self.clean_after_initialization(json_path)

        # Initialize the route space
        self.update_route_space()

    #        print(self.route_space['C->A->D'])
    #        print(self.nodes['A'].switching_matrix['C']['D'])

    # Generates for all the possible paths in weighted paths a list of all the nodes
    # and a list of all the lines belonging to that path
    def generate_node_and_line_list(self):
        for i in range(len(self.weighted_paths['path'])):
            current_path = self.weighted_paths['path'][i]

            # Generates the list of the lines in the best path
            line_list = list()
            node_list = list()
            for index in range(int(((len(current_path) - 4) / 3)) + 1):
                line_list.append(current_path[3 * index] + current_path[3 * index + 3])
                node_list.append(current_path[3 * index])
            node_list.append(current_path[-1])

            self.line_list_dict[i] = line_list
            self.node_list_dict[i] = node_list

    # Cleans the node switching matrix and line states
    def clean_after_initialization(self, json_path):
        # Cleans line states
        for line_label in self.lines:
            self.lines[line_label].state[0] = 1

        node_json = json.load(open(json_path, 'r'))
        for node_label in node_json:
            # Create the node instance
            node_dict = node_json[node_label]
#            node_dict['label'] = node_label
#            node = Node(node_dict)
#            self._nodes[node_label] = node
            self.switching_matrix_dict[node_label] = node_dict['switching_matrix']
            for connected_node1_label in self.switching_matrix_dict[node_label]:
                for connected_node2_label in self.switching_matrix_dict[node_label][connected_node1_label]:
                    self.switching_matrix_dict[node_label][connected_node1_label][connected_node2_label] = \
                        np.array(self.switching_matrix_dict[node_label][connected_node1_label][connected_node2_label])

        nodes_dict = self.nodes
        for node_label in nodes_dict:
            node = nodes_dict[node_label]
            for connected_node in node.connected_nodes:
                # Initializes switching matrix
                node.switching_matrix[connected_node] = self.switching_matrix_dict[node_label][connected_node]

    # Draws the topology of the network
    def draw(self):
        nodes = self.nodes

        for node_label in nodes:
            n0 = nodes[node_label]
            x0 = n0.position[0]
            y0 = n0.position[1]
            plt.plot(x0, y0, 'go', markersize=10)
            plt.text(x0 + 20, y0 + 20, node_label)
            for connected_node_label in n0.connected_nodes:
                n1 = nodes[connected_node_label]
                x1 = n1.position[0]
                y1 = n1.position[1]
                plt.plot([x0, x1], [y0, y1], 'b')
        plt.title('Network ')
        plt.show()

    # Finds all the possible paths between two nodes
    def find_paths(self, label1, label2):
        cross_nodes = [key for key in self.nodes.keys()
                       if ((key != label1) & (key != label2))]
        cross_lines = self.lines.keys()
        inner_paths = {'0': label1}
        for i in range(len(cross_nodes)+1):
            inner_paths[str(i+1)] = []
            for inner_path in inner_paths[str(i)]:
                inner_paths[str(i+1)] += [inner_path + cross_node
                                          for cross_node in cross_nodes
                                          if ((inner_path[-1] + cross_node in cross_lines) &
                                              (cross_node not in inner_path))]
        paths = []
        for i in range(len(cross_nodes)+1):
            for path in inner_paths[str(i)]:
                if path[-1] + label2 in cross_lines:
                    paths.append(path + label2)
        return paths

    # For each node it finds the connected lines and vice versa
    # Creates the switching matrix for each node
    def connect(self):
        nodes_dict = self.nodes
        lines_dict = self.lines
        for node_label in nodes_dict:
            node = nodes_dict[node_label]
            for connected_node in node.connected_nodes:
                line_label = node_label + connected_node
                line = lines_dict[line_label]
                line.successive[connected_node] = nodes_dict[connected_node]
                node.successive[line_label] = lines_dict[line_label]

                # Initializes switching matrix
                node.switching_matrix[connected_node] = self.switching_matrix_dict[node_label][connected_node]

    # Propagates the signal information along the path
    def propagate(self, signal_information):
        path = signal_information.path
        start_node = self.nodes[path[0]]
        propagated_signal_information = start_node.propagate(signal_information)
        return propagated_signal_information

    # Finds the path between two nodes with the best snr
    def find_best_snr(self, node_input, node_output):
        best_snr = -math.inf
        best_index = -1
        free_channel_index = -1

        for i in range(len(self.weighted_paths['path'])):
            current_path = self.weighted_paths['path'][i]
            # For each possible path, checks if the input node and the output node corresponds to the right ones
            if (current_path[0] == node_input) and (current_path[-1] == node_output):
                for channel in range(number_of_channels):
                    # For each channel of the current path, if a channel is free for all the lines in the current path,
                    # the best snr and best index are updated if the current snr is better than the old one
                    if self.route_space[current_path][channel] == 1:
                        if self.weighted_paths['snr'][i] > best_snr:
                            best_snr = self.weighted_paths['snr'][i]
                            best_index = i
                            free_channel_index = channel
                        break

        find_best_snr_return = [best_index, free_channel_index]
        return find_best_snr_return

    # Finds the path between two nodes with the best latency
    def find_best_latency(self, node_input, node_output):
        best_latency = +math.inf
        best_index = -1
        free_channel_index = -1

        for i in range(len(self.weighted_paths['path'])):
            current_path = self.weighted_paths['path'][i]

            # For each possible path, checks if the input node and the output node corresponds to the right ones
            if (current_path[0] == node_input) and (current_path[-1] == node_output):
                for channel in range(number_of_channels):
                    # For each channel of the current path, if a channel is free for all the lines in the current path,
                    # the best latency and best index are updated if the current latency is better than the old one
                    if self.route_space[current_path][channel] == 1:
                        if self.weighted_paths['latency'][i] < best_latency:
                            best_latency = self.weighted_paths['latency'][i]
                            best_index = i
                            free_channel_index = channel
                        break

        find_best_snr_return = [best_index, free_channel_index]
        return find_best_snr_return

    # Updates the route space
    def update_route_space(self):
        route_space_dict = {}

        for i in range(len(self.weighted_paths['path'])):

            line_list = self.line_list_dict[i]
            node_list = self.node_list_dict[i]

            # For each path, for each channel, a blocking event is generated if
            # 1. one of the switching matrix of the nodes in the path is occupied
            # 2. one of the lines in the path is occupied
            block = np.ones(number_of_channels)
            for channel in range(number_of_channels):
                for line in line_list:
                    block[channel] = block[channel] * self.lines[line].state[channel]
#                    if block[channel] == 0:
#                        break
                for node_index in range(1, len(node_list)-1):
                    block[channel] = block[channel] * self.nodes[node_list[node_index]].\
                        switching_matrix[node_list[node_index-1]][node_list[node_index+1]][channel]
#                    if block[channel] == 0:
#                        break
            route_space_dict[self.weighted_paths['path'][i]] = block

        self.route_space = pd.DataFrame(route_space_dict)

    # Generates the connection from the connection list
    # Finds the path with the best snr/latency
    def stream(self, connection_list, snr_or_latency='latency'):
        best_path_index_list = []
        if snr_or_latency == 'snr':
            for i in range(len(connection_list)):
                find_best_snr_output = Network.find_best_snr(self, connection_list[i].input_node,
                                                             connection_list[i].output_node)
                best_path_index_list.append(find_best_snr_output[0])
                free_channel = find_best_snr_output[1]
                bit_rate = 0

                # Bit-rate check
                if best_path_index_list[i] != -1:
                    first_node = self.path_list[best_path_index_list[i]][0]
                    bit_rate = self.calculate_bit_rate(best_path_index_list[i], self.nodes[first_node].transceiver)
                    if bit_rate == 0:
                        best_path_index_list[i] = -1

                # If the path is found, updates the connection class
                if best_path_index_list[i] != -1:
                    deployed_lightpath = Lightpath(input_signal_power,
                                                   self.path_list[best_path_index_list[i]], free_channel)

                    deployed_lightpath = self.propagate(deployed_lightpath)

                    connection_list[i].bit_rate = bit_rate
                    connection_list[i].latency = deployed_lightpath.latency
                    connection_list[i].snr = 10 * np.log10(
                        deployed_lightpath.signal_power / deployed_lightpath.noise_power)

                    self.update_route_space()

                # If no path is found, sets latency to 0 and snr to None
                else:
                    connection_list[i].latency = 0
                    connection_list[i].snr = None

        elif snr_or_latency == 'latency':
            for i in range(len(connection_list)):
                find_best_latency_output = Network.find_best_latency(self, connection_list[i].input_node,
                                                                      connection_list[i].output_node)
                best_path_index_list.append(find_best_latency_output[0])
                free_channel = find_best_latency_output[1]
                # If the path is found, updates the connection class
                if best_path_index_list[i] != -1:
                    deployed_lightpath = Lightpath(input_signal_power,
                                                   self.path_list[best_path_index_list[i]], free_channel)
                    deployed_lightpath = self.propagate(deployed_lightpath)
                    connection_list[i].latency = deployed_lightpath.latency
                    connection_list[i].snr = 10 * np.log10(
                        deployed_lightpath.signal_power / deployed_lightpath.noise_power)

                    self.update_route_space()

                # If no path is found, sets latency to 0 and snr to None
                else:
                    connection_list[i].latency = 0
                    connection_list[i].snr = None

        else:
            print('Choice not valid')

    # Calculates the bit rate of the path based on the strategy choice
    def calculate_bit_rate(self, path, strategy):

        Rb = 0
        gsnrdB = self.weighted_paths['snr'][path]

        gsnr = 10 ** (gsnrdB / 10)
        if strategy == 'fixed_rate':
            if gsnr >= 2*(special.erfinv(2*BERt)**2)*(Rs/Bn):
                Rb = 100
            else:
                Rb = 0
        elif strategy == 'flex_rate':
            if gsnr < 2*(special.erfinv(2*BERt)**2)*(Rs/Bn):
                Rb = 0
            elif gsnr < (14/3)*(special.erfinv((3/2)*BERt)**2)*(Rs/Bn):
                Rb = 100
            elif gsnr < 10*(special.erfinv((8/3)*BERt)**2)*(Rs/Bn):
                Rb = 200
            else:
                Rb = 400
        elif strategy == 'shannon':
            Rb = (2 * Rs * math.log2(1 + gsnr * (Bn / Rs))) * (10 ** (-9))
        else:
            print('Strategy non valid')
            exit()

        return Rb


# Generates the connection between two nodes: contains the input and output nodes, the signal power and the latency/snr
# of the chosen path
class Connection:
    def __init__(self, input_node, output_node, signal_power):
        self._input_node = input_node
        self._output_node = output_node
        self._signal_power = signal_power
        self._latency = 0.0
        self._snr = 0.0
        self._bit_rate = 0

    @property
    def input_node(self):
        return self._input_node

    @property
    def output_node(self):
        return self._output_node

    @property
    def signal_power(self):
        return self._signal_power

    @property
    def latency(self):
        return self._latency

    @latency.setter
    def latency(self, latency):
        self._latency = latency

    @property
    def snr(self):
        return self._snr

    @snr.setter
    def snr(self, snr):
        self._snr = snr

    @property
    def bit_rate(self):
        return self._bit_rate

    @bit_rate.setter
    def bit_rate(self, bit_rate):
        self._bit_rate = bit_rate


def main():

    # Initialize an object network of class Network
    network = Network(json_path1)

    # Fills "successive" attributes of Nodes and Lines
    network.connect()

    # Fills weighted paths and initialize route_space attributes
    network.initialize(json_path1)

    # Input/Output generation
    input_node = []
    output_node = []
    for i in range(number_of_connections):
        temp_in = rnd.randint(0, 5)
        while True:
            temp_out = rnd.randint(0, 5)
            if temp_out != temp_in:
                break
        number_to_node = ['A', 'B', 'C', 'D', 'E', 'F']
        input_node.append(number_to_node[temp_in])
        output_node.append(number_to_node[temp_out])

    # Connection generation
    connection_list = []
    for i in range(len(input_node)):
        connection_list.append(Connection(input_node[i], output_node[i], input_signal_power))

    # Stream call
    network.stream(connection_list, snr_or_latency_choice)
    snr_list = list()
    latency_list = list()
    bit_rate_list = list()

    # Result printing
    for i in range(len(input_node)):
        snr_list.append(connection_list[i].snr)
        latency_list.append(connection_list[i].latency)
        bit_rate_list.append(connection_list[i].bit_rate)

    number_of_blocks_full_fixed = 0
    snr_list_no_none = []
    latency_no_zero = []
    bit_rate_no_zero = []
    for index in range(len(snr_list)):
        if snr_list[index] is not None:
            snr_list_no_none.append(snr_list[index])
        else:
            number_of_blocks_full_fixed = number_of_blocks_full_fixed + 1
        if latency_list[index] != 0:
            latency_no_zero.append(latency_list[index])
        if bit_rate_list[index] != 0:
            bit_rate_no_zero.append(bit_rate_list[index])

    # Conversion to array for plotting
    snr_array = np.array(snr_list_no_none)
    latency_array = np.array(latency_no_zero)
    bit_rate_array = np.array(bit_rate_no_zero)

    # 2nd NETWORK ################################################################

    # Initialize an object network of class Network
    network2 = Network(json_path2)

    # Fills "successive" attributes of Nodes and Lines
    network2.connect()

    # Fills weighted paths and initialize route_space attributes
    network2.initialize(json_path2)

    # Connection generation
    connection_list2 = []
    for i in range(len(input_node)):
        connection_list2.append(Connection(input_node[i], output_node[i], input_signal_power))

    # Stream call
    network2.stream(connection_list2, snr_or_latency_choice)
    snr_list2 = list()
    latency_list2 = list()
    bit_rate_list2 = list()

    # Result printing
    for i in range(len(input_node)):
        snr_list2.append(connection_list2[i].snr)
        latency_list2.append(connection_list2[i].latency)
        bit_rate_list2.append(connection_list2[i].bit_rate)

    number_of_blocks_full_flex = 0
    snr_list_no_none2 = []
    latency_no_zero2 = []
    bit_rate_no_zero2 = []

    for index in range(len(snr_list2)):
        if snr_list2[index] is not None:
            snr_list_no_none2.append(snr_list2[index])
        else:
            number_of_blocks_full_flex = number_of_blocks_full_flex + 1
        if latency_list2[index] != 0:
            latency_no_zero2.append(latency_list2[index])
        if bit_rate_list2[index] != 0:
            bit_rate_no_zero2.append(bit_rate_list2[index])

    # Conversion to array for plotting
    snr_array2 = np.array(snr_list_no_none2)
    latency_array2 = np.array(latency_no_zero2)
    bit_rate_array2 = np.array(bit_rate_no_zero2)

    # 3rd NETWORK ###################################################################

    # Initialize an object network of class Network
    network3 = Network(json_path3)

    # Fills "successive" attributes of Nodes and Lines
    network3.connect()

    # Fills weighted paths and initialize route_space attributes
    network3.initialize(json_path3)

    # Connection generation
    connection_list3 = []
    for i in range(len(input_node)):
        connection_list3.append(Connection(input_node[i], output_node[i], input_signal_power))

    # Stream call
    network3.stream(connection_list3, snr_or_latency_choice)
    snr_list3 = list()
    latency_list3 = list()
    bit_rate_list3 = list()

    # Result printing
    for i in range(len(input_node)):
        snr_list3.append(connection_list3[i].snr)
        latency_list3.append(connection_list3[i].latency)
        bit_rate_list3.append(connection_list3[i].bit_rate)

    number_of_blocks_full_shannon = 0
    snr_list_no_none3 = []
    latency_no_zero3 = []
    bit_rate_no_zero3 = []
    for index in range(len(snr_list3)):
        if snr_list3[index] is not None:
            snr_list_no_none3.append(snr_list3[index])
        else:
            number_of_blocks_full_shannon = number_of_blocks_full_shannon + 1
        if latency_list3[index] != 0:
            latency_no_zero3.append(latency_list3[index])
        if bit_rate_list3[index] != 0:
            bit_rate_no_zero3.append(bit_rate_list3[index])

    # Conversion to array for plotting
    snr_array3 = np.array(snr_list_no_none3)
    latency_array3 = np.array(latency_no_zero3)
    bit_rate_array3 = np.array(bit_rate_no_zero3)

    print('Blocking events for full fixed rate = ', number_of_blocks_full_fixed)
    print('Blocking events for full flex rate = ', number_of_blocks_full_flex)
    print('Blocking events for full shannon = ', number_of_blocks_full_shannon)

    bit_rate_mean = np.mean(bit_rate_array)
    bit_rate_mean2 = np.mean(bit_rate_array2)
    bit_rate_mean3 = np.mean(bit_rate_array3)
    print('Bit rate mean for full fixed rate = ', bit_rate_mean, 'Gbps')
    print('Bit rate mean for full flex rate = ', bit_rate_mean2, 'Gbps')
    print('Bit rate mean for full shannon = ', bit_rate_mean3, 'Gbps')

    capacity = np.sum(bit_rate_array)
    capacity2 = np.sum(bit_rate_array2)
    capacity3 = np.sum(bit_rate_array3)
    print('Capacity for full fixed rate = ', capacity, 'Gbps')
    print('Capacity for full flex rate = ', capacity2, 'Gbps')
    print('Capacity for full shannon = ', capacity3, 'Gbps')

    # Result plotting

#    plt.hist(snr_array, color='blue', edgecolor='black', bins=50)
#    plt.show()
#    plt.hist(latency_array, color='blue', edgecolor='black', bins=50)
#    plt.show()
#    plt.hist(bit_rate_array, color='blue', edgecolor='black', bins=50)
#    plt.show()

#    plt.hist(snr_array2, color='blue', edgecolor='black', bins=50)
#    plt.show()
#    plt.hist(latency_array2, color='blue', edgecolor='black', bins=50)
#    plt.show()
#    plt.hist(bit_rate_array2, color='blue', edgecolor='black', bins=50)
#    plt.show()

#    plt.hist(snr_array3, color='blue', edgecolor='black', bins=50)
#    plt.show()
#    plt.hist(latency_array3, color='blue', edgecolor='black', bins=50)
#    plt.show()
#    plt.hist(bit_rate_array3, color='blue', edgecolor='black', bins=50)
#    plt.show()


if __name__ == '__main__':
    main()
