import numpy as np

from Core.elements import Network

from Core.utils import json_path1
from Core.utils import json_path2
from Core.utils import json_path3
from Core.utils import snr_or_latency_choice
from Core.utils import print_input_matrix
from Core.utils import print_output_matrix
from Core.utils import write_output_file
from Core.utils import plot_results
from Core.utils import generate_requests
from Core.utils import generate_output_arrays
from Core.utils import update_congestion
from Core.utils import plot_congestion
from Core.utils import update_capacity
from Core.utils import plot_capacity
from Core.utils import update_bit_rate_shannon
from Core.utils import plot_bit_rate_shannon

from pathlib import Path

root = Path(__file__).parent.parent


def main():
    average_congestion_network1 = []
    average_congestion_network2 = []
    average_congestion_network3 = []
    capacity_network1 = []
    capacity_network2 = []
    capacity_network3 = []
    average_bit_rate_shannon = []
    for M in range(1, 11):

        # Initialize an object network of class Network
        network = Network(root / json_path1)

        # Fills "successive" attributes of Nodes and Lines
        network.connect()

        # Fills weighted paths and initialize route_space attributes
        network.initialize(root / json_path1)

        # Input/Output and requests generation
        generate_requests_output = generate_requests()
        input_node = generate_requests_output[0]
        output_node = generate_requests_output[1]

        # Generate traffic matrix
        traffic_matrix = network.generate_traffic_matrix(input_node, output_node, M)
        node_list = ['A', 'B', 'C', 'D', 'E', 'F']

        # writes the input traffic matrix to the output file
        traffic_matrix_mat = np.asmatrix(traffic_matrix)
        input_traffic_matrix_path = 'Results/Lab10/input_traffic_matrix_M_' + str(M) + '.png'
        print_input_matrix(traffic_matrix_mat, node_list, root, input_traffic_matrix_path)

        # Connection generation
        connections_management_output = network.connections_management(traffic_matrix, node_list, snr_or_latency_choice)
        traffic_matrix = connections_management_output[0]
        connection_list = connections_management_output[1]

        # writes the output matrix to the output file
        traffic_matrix_mat = np.asmatrix(traffic_matrix)
        traffic_matrix_mat = network.handle_output_traffic_matrix(traffic_matrix_mat)
        output_traffic_matrix_1_path = 'Results/Lab10/output_traffic_matrix_fixed_M_' + str(M) + '.png'
        print_output_matrix(traffic_matrix_mat, node_list, root, output_traffic_matrix_1_path)

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

        traffic_matrix2 = network2.generate_traffic_matrix(input_node, output_node, M)

        # Connection generation
        connections_management_output2 = network2.connections_management(traffic_matrix2, node_list,
                                                                         snr_or_latency_choice)
        traffic_matrix2 = connections_management_output2[0]
        connection_list2 = connections_management_output2[1]

        # writes the output matrix to the output file
        traffic_matrix_mat2 = np.asmatrix(traffic_matrix2)
        traffic_matrix_mat2 = network2.handle_output_traffic_matrix(traffic_matrix_mat2)
        output_traffic_matrix_2_path = 'Results/Lab10/output_traffic_matrix_flex_M_' + str(M) + '.png'
        print_output_matrix(traffic_matrix_mat2, node_list, root, output_traffic_matrix_2_path)

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

        traffic_matrix3 = network3.generate_traffic_matrix(input_node, output_node, M)

        # Connection generation
        connections_management_output3 = network3.connections_management(traffic_matrix3, node_list,
                                                                         snr_or_latency_choice)
        traffic_matrix3 = connections_management_output3[0]
        connection_list3 = connections_management_output3[1]

        # writes the output matrix to the output file
        traffic_matrix_mat3 = np.asmatrix(traffic_matrix3)
        traffic_matrix_mat3 = network3.handle_output_traffic_matrix(traffic_matrix_mat3)
        output_traffic_matrix_3_path = 'Results/Lab10/output_traffic_matrix_shannon_M_' + str(M) + '.png'
        print_output_matrix(traffic_matrix_mat3, node_list, root, output_traffic_matrix_3_path)

        # Generates snr, latency, bit rate for plotting and computes the number of blocking events
        generate_output_arrays_output3 = generate_output_arrays(connection_list3)
        snr_array3 = generate_output_arrays_output3[0]
        latency_array3 = generate_output_arrays_output3[1]
        bit_rate_array3 = generate_output_arrays_output3[2]
        number_of_blocks_full_shannon = generate_output_arrays_output3[3]

        # Write bit rates, capacities and blocking events in output file
        write_output_file(M, root, number_of_blocks_full_fixed, number_of_blocks_full_flex,
                          number_of_blocks_full_shannon, bit_rate_array, bit_rate_array2, bit_rate_array3)

        # Result plotting
        plot_results(M, root, snr_array, latency_array, bit_rate_array, snr_array2, latency_array2, bit_rate_array2,
                     snr_array3, latency_array3, bit_rate_array3)

        update_congestion(network, network2, network3, average_congestion_network1,
                          average_congestion_network2, average_congestion_network3)
        update_capacity(capacity_network1, capacity_network2, capacity_network3, bit_rate_array, bit_rate_array2,
                        bit_rate_array3)
        update_bit_rate_shannon(average_bit_rate_shannon, bit_rate_array3)

    plot_congestion(root, average_congestion_network1, average_congestion_network2, average_congestion_network3)
    plot_capacity(root, capacity_network1, capacity_network2, capacity_network3)
    plot_bit_rate_shannon(root, average_bit_rate_shannon)


if __name__ == '__main__':
    main()
