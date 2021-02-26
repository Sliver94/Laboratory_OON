import json
import numpy as np

c = 2/3*(3 * (10 ** 8)) #speed of light


class signal_information:
    def __init__(self, signal_power, path):
        self._signal_power = signal_power
        self._noise_power = 0.0
        self._latency = 0.0
        self._path = path

    def get_signal_power(self):
        return self._signal_power

    def set_signal_power(self, signal_power):
        self._signal_power = signal_power

    def get_path(self):
        return self._path

    def set_path(self, path):
        self._path = path

    def increment_signal(self, inc_signal_power):
        self._signal_power += inc_signal_power

    def get_noise_power(self):
        return self._noise_power

    def increment_noise(self, inc_noise_power):
        self._noise_power += inc_noise_power

    def increment_latency(self, inc_latency):
        self._latency += inc_latency

    def path_eval(self):
        self._path.pop(0)
        return self._path[0]


class Node:
    def __init__(self, values):
        self.label = values['label']
        self.position = values['position']
        self.connected_nodes = values['connected_nodes']
        self.successive = {}
        self.next_node = '_'

    def propagate_node(self, signal_information):
        self.next_node = signal_information.path_eval()
        return self.next_node


class Line:
    def __init__(self, label, length):
        self.label = label
        self.length = length
        self.successive = {}

    def propagate_line(self, signal_information):
        signal_information.increment_noise(self.noise_generation(signal_information.signal_power))
        signal_information.increment_latency(self.latency_generation())

    def latency_generation(self):
        return self.length*c

    def noise_generation(self, signal_power):
        return 1e-3*signal_power*self.length


class Network:
    def __init__(self):
        with open('nodes.json', 'r') as f:
            self.nodes_json = json.load(f)
        lines = {}
        nodes = {}
        for x in self.nodes_json:
            connected_nodes = self.nodes_json[x]['connected_nodes']
            for i in range(len(connected_nodes)):
                if x != connected_nodes[i]:
                    dist = np.sqrt((self.nodes_json[x]['position'][0] -
                                    self.nodes_json[connected_nodes[i]]['position'][0])**2 +
                                   (self.nodes_json[x]['position'][1]-
                                    self.nodes_json[connected_nodes[i]]['position'][1])**2)
                    lines[x+connected_nodes[i]] = Line(x+connected_nodes[i], dist)
            values = {"label": x, "position": self.nodes_json[x]['position'], "connected_nodes": connected_nodes}
            nodes[x] = Node(values)
        self.lines = lines
        self.nodes = nodes

    def connect(self):
        for node_name in self.nodes_json:
            connected_nodes = self.nodes[node_name].connected_nodes
            for i in range(len(connected_nodes)):
                self.nodes[node_name].successive[node_name+connected_nodes[i]] = \
                    self.lines[node_name+connected_nodes[i]]


net = Network()

net.connect()

N = net.nodes
L = net.lines






