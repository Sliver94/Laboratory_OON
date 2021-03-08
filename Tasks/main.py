from Core.elements import Network
from Core.parameters import Connection

from Core.utils import json_path1
from Core.utils import json_path2
from Core.utils import json_path3
from Core.utils import input_signal_power
from Core.utils import snr_or_latency_choice
from Core.utils import generate_requests
from Core.utils import generate_output_arrays
from Core.utils import plot_results

from pathlib import Path

root = Path(__file__).parent.parent


def main():
    # Initialize an object network of class Network
    network = Network(root / json_path1)

    # Fills "successive" attributes of Nodes and Lines
    network.connect()

    # Fills weighted paths and initialize route_space attributes
    network.initialize(root / json_path1)

    # Input/Output generation
    generate_requests_output = generate_requests()
    input_node = generate_requests_output[0]
    output_node = generate_requests_output[1]

    # Connection generation
    connection_list = []
    for i in range(len(input_node)):
        connection_list.append(Connection(input_node[i], output_node[i], input_signal_power))

    # Stream call
    network.stream(connection_list, snr_or_latency_choice)

    # Generates snr, latency, bit rate for plotting and computes the number of blocking events
    generate_output_arrays_output = generate_output_arrays(connection_list)
    snr_array = generate_output_arrays_output[0]
    latency_array = generate_output_arrays_output[1]
    bit_rate_array = generate_output_arrays_output[2]
    number_of_blocks_full_fixed = generate_output_arrays_output[3]

    # 2nd NETWORK ################################################################

    # Initialize an object network of class Network
    network2 = Network(root / json_path2)

    # Fills "successive" attributes of Nodes and Lines
    network2.connect()

    # Fills weighted paths and initialize route_space attributes
    network2.initialize(root / json_path2)

    # Connection generation
    connection_list2 = []
    for i in range(len(input_node)):
        connection_list2.append(Connection(input_node[i], output_node[i], input_signal_power))

    # Stream call
    network2.stream(connection_list2, snr_or_latency_choice)

    # Generates snr, latency, bit rate for plotting and computes the number of blocking events
    generate_output_arrays_output2 = generate_output_arrays(connection_list2)
    snr_array2 = generate_output_arrays_output2[0]
    latency_array2 = generate_output_arrays_output2[1]
    bit_rate_array2 = generate_output_arrays_output2[2]
    number_of_blocks_full_flex = generate_output_arrays_output2[3]

    # 3rd NETWORK ###################################################################

    # Initialize an object network of class Network
    network3 = Network(root / json_path3)

    # Fills "successive" attributes of Nodes and Lines
    network3.connect()

    # Fills weighted paths and initialize route_space attributes
    network3.initialize(root / json_path3)

    # Connection generation
    connection_list3 = []
    for i in range(len(input_node)):
        connection_list3.append(Connection(input_node[i], output_node[i], input_signal_power))

    # Stream call
    network3.stream(connection_list3, snr_or_latency_choice)

    # Generates snr, latency, bit rate for plotting and computes the number of blocking events
    generate_output_arrays_output3 = generate_output_arrays(connection_list3)
    snr_array3 = generate_output_arrays_output3[0]
    latency_array3 = generate_output_arrays_output3[1]
    bit_rate_array3 = generate_output_arrays_output3[2]
    number_of_blocks_full_shannon = generate_output_arrays_output3[3]

    plot_results(root, number_of_blocks_full_fixed, number_of_blocks_full_flex, number_of_blocks_full_shannon,
                 bit_rate_array, bit_rate_array2, bit_rate_array3, snr_array, snr_array2, snr_array3, latency_array,
                 latency_array2, latency_array3)


if __name__ == '__main__':
    main()
