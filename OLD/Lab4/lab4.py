import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rnd

c = 3 * (10 ** 8)                # Speed of light
c_network = 2/3 * c              # Speed of signal on the network
json_path_global = 'nodes.json'  # Json file address


class SignalInformation:
    def __init__(self, power, path):
        self._signal_power = power
        self._path = path
        self._noise_power = 0.0
        self._latency = 0.0

    @property
    def signal_power(self):
        return self._signal_power

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
    def noise_power(self, noise):
        self._noise_power = noise

    @property
    def latency(self):
        return self._latency

    @latency.setter
    def latency(self, latency):
        self._latency = latency

    def add_noise(self, noise):
        self.noise_power += noise

    def add_latency(self, latency):
        self.latency += latency

    def next(self):
        self.path = self.path[1:]


class Node:
    def __init__(self, node_dict):
        self._label = node_dict['label']
        self._position = node_dict['position']
        self._connected_nodes = node_dict['connected_nodes']
        self._successive = {}

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

    def propagate(self, signal_information):
        path = signal_information.path
        if len(path) > 1:
            line_label = path[:2]
            line = self.successive[line_label]
            signal_information.next()
            signal_information = line.propagate(signal_information)
        return signal_information


class Line:
    def __init__(self, line_dict):
        self._label = line_dict['label']
        self._length = line_dict['length']
        self._successive = {}
        self._state = 'free'

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

    def latency_generation(self):
        latency = self.length / c_network
        return latency

    def noise_generation(self, signal_power):
        noise = 1e-3 * signal_power * self.length
        return noise

    def propagate(self, signal_information):
        # Update latency
        latency = self.latency_generation()
        signal_information.add_latency(latency)

        # Update noise
        signal_power = signal_information.signal_power
        noise = self.noise_generation(signal_power)
        signal_information.add_noise(noise)

        node = self.successive[signal_information.path[0]]
        signal_information = node.propagate(signal_information)
        return signal_information


class Network:
    def __init__(self, json_path, df):
        self._weighted_paths = df
        node_json = json.load(open(json_path, 'r'))
        self._nodes = {}
        self._lines = {}
        for node_label in node_json:
            # Create the node instance
            node_dict = node_json[node_label]
            node_dict['label'] = node_label
            node = Node(node_dict)
            self._nodes[node_label] = node

            # Create the line instances
            for connected_node_label in node_dict['connected_nodes']:
                line_dict = {}
                line_label = node_label + connected_node_label
                line_dict['label'] = line_label
                node_position = np.array(node_json[node_label]['position'])
                connected_node_position = np.array(node_json[connected_node_label]['position'])
                line_dict['length'] = np.sqrt(np.sum((node_position-connected_node_position)**2))
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

    def propagate(self, signal_information):
        path = signal_information.path
        start_node = self.nodes[path[0]]
        propagated_signal_information = start_node.propagate(signal_information)
        return propagated_signal_information

    # Modificare find_best_snr e find_best_latency: se trova un path libero, lo occupa.
    # Se ne trova uno migliore, libera quello di prima e ne occupa uno nuovo.
    # Se non trova path liberi, deve tornare snr=zero e latency=none
    def find_best_snr(self, node_input, node_output):

        best_snr = 0
        best_index = 0
        first_cycle = True

        for i in range(len(self.weighted_paths['path'])):
            current_path = self.weighted_paths['path'][i]
            if (current_path[0] == node_input) and (current_path[-1] == node_output):
                current_snr = self.weighted_paths['snr'][i]
                if first_cycle:
                    best_snr = current_snr
                    first_cycle = False
                    best_index = i
                elif current_snr > best_snr:
                    best_snr = current_snr
                    best_index = i
        return best_index

    def find_best_latency(self, node_input, node_output):
        best_latency = 0
        best_index = 0
        first_cycle = True

        for i in range(len(self.weighted_paths['path'])):
            current_path = self.weighted_paths['path'][i]
            if (current_path[0] == node_input) and (current_path[-1] == node_output):
                current_latency = self.weighted_paths['latency'][i]
                if first_cycle:
                    best_latency = current_latency
                    first_cycle = False
                    best_index = i
                elif current_latency < best_latency:
                    best_latency = current_latency
                    best_index = i
        return best_index

    def stream(self, connection_list, snr_or_latency_choice='latency'):
        best_path_index_list = []
        if snr_or_latency_choice == 'snr':
            for i in range(len(connection_list)):
                best_path_index_list.append(Network.find_best_snr(self, connection_list[i].input_node,
                                                                  connection_list[i].output_node))
                connection_list[i].latency = self.weighted_paths['latency'][best_path_index_list[i]]
                connection_list[i].snr = self.weighted_paths['snr'][best_path_index_list[i]]

        elif snr_or_latency_choice == 'latency':
            for i in range(len(connection_list)):
                best_path_index_list.append(Network.find_best_latency(self, connection_list[i].input_node,
                                                                      connection_list[i].output_node))
                connection_list[i].latency = self.weighted_paths['latency'][best_path_index_list[i]]
                connection_list[i].snr = self.weighted_paths['snr'][best_path_index_list[i]]

        else:
            print('Choice not valid')


class Connection:
    def __init__(self, input_node, output_node, signal_power):
        self._input_node = input_node
        self._output_node = output_node
        self._signal_power = signal_power
        self._latency = 0.0
        self._snr = 0.0

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


def main():

    # Data frame generation
    df = pd.DataFrame()
    network = Network(json_path_global, df)
    network.connect()
    node_labels = network.nodes.keys()
    pairs = []
    for label1 in node_labels:
        for label2 in node_labels:
            if label1 != label2:
                pairs.append(label1+label2)
    # Columns = ['path', 'latency', 'noise', 'snr']
    paths = []
    latencies = []
    noises = []
    snrs = []
    for pair in pairs:
        for path in network.find_paths(pair[0], pair[1]):
            path_string = ''
            for node in path:
                path_string += node + '->'
            paths.append(path_string[:-2])

            # Propagation
            signal_information = SignalInformation(0.001, path)
            signal_information = network.propagate(signal_information)
            latencies.append(signal_information.latency)
            noises.append(signal_information.noise_power)
            snrs.append(
                10*np.log10(
                    signal_information.signal_power/signal_information.noise_power
                )
            )
    df['path'] = paths
    df['latency'] = latencies
    df['noise'] = noises
    df['snr'] = snrs
    network.weighted_paths = df
    print(df)
    # Input/Output generation
    input_node = []
    output_node = []
    for i in range(100):

        temp_in = rnd.randint(0, 5)
        while True:
            temp_out = rnd.randint(0, 5)
            if temp_out != temp_in:
                break
        if temp_in == 0:
            input_node.append('A')
        elif temp_in == 1:
            input_node.append('B')
        elif temp_in == 2:
            input_node.append('C')
        elif temp_in == 3:
            input_node.append('D')
        elif temp_in == 4:
            input_node.append('E')
        elif temp_in == 5:
            input_node.append('F')
        if temp_out == 0:
            output_node.append('A')
        elif temp_out == 1:
            output_node.append('B')
        elif temp_out == 2:
            output_node.append('C')
        elif temp_out == 3:
            output_node.append('D')
        elif temp_out == 4:
            output_node.append('E')
        elif temp_out == 5:
            output_node.append('F')

    # Connection generation
    connection_list = []
    for i in range(len(input_node)):
        connection_list.append(Connection(input_node[i], output_node[i], 0.001))

    # Stream call
    network.stream(connection_list, 'snr')
    snr_list = list()
    latency_list = list()

    print('Best snr case')
    for i in range(len(input_node)):
        print('For input = ', input_node[i], ', output = ', output_node[i], ', the SNR is: ',
              connection_list[i].snr)
        snr_list.append(connection_list[i].snr)

    snr_array = np.array(snr_list)

    plt.hist(snr_array, color='blue', edgecolor='black',
             bins=30)
    plt.show()

    # Stream call
    network.stream(connection_list, 'latency')

    print('Best latency case')
    for i in range(len(input_node)):
        print('For input = ', input_node[i], ', output = ', output_node[i], ', the latency is: ',
              connection_list[i].latency)
        latency_list.append(connection_list[i].latency)

    latency_array = np.array(latency_list)

    plt.hist(latency_array, color='blue', edgecolor='black',
             bins=30)
    plt.show()


if __name__ == '__main__':
    main()
