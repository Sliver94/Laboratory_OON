import pandas as pd

from Core.elements import SignalInformation
from Core.elements import Network
from Core.utils import json_path
from Core.science_utils import linear_to_db_conversion

from pathlib import Path
root = Path(__file__).parent.parent


def main():
    network = Network(root / json_path)
    network.connect()
    node_labels = network.nodes.keys()
    pairs = []
    for label1 in node_labels:
        for label2 in node_labels:
            if label1 != label2:
                pairs.append(label1+label2)

#    columns = ['path', 'latency', 'noise', 'snr']
    df = pd.DataFrame()
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
                linear_to_db_conversion(
                    signal_information.signal_power/signal_information.noise_power
                )
            )
    df['path'] = paths
    df['latency'] = latencies
    df['noise'] = noises
    df['snr'] = snrs
    print(df)
    network.draw()


if __name__ == '__main__':
    main()
