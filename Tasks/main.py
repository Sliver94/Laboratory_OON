import random as rnd
import numpy as np
import matplotlib.pyplot as plt

from Core.elements import Network
from Core.science_utils import generate_traffic_matrix
from Core.utils import json_path1
from Core.utils import json_path2
from Core.utils import json_path3
from Core.utils import input_signal_power
from Core.utils import number_of_connections
from Core.utils import snr_or_latency_choice
from Core.parameters import Connection

from pathlib import Path

root = Path(__file__).parent.parent


def main():

    for M in range(1, 6):

        # Initialize an object network of class Network
        network = Network(root / json_path1)

        # Fills "successive" attributes of Nodes and Lines
        network.connect()

        # Fills weighted paths and initialize route_space attributes
        network.initialize(root / json_path1)

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

        traffic_matrix = generate_traffic_matrix(input_node, output_node, M)
        node_list = ['A', 'B', 'C', 'D', 'E', 'F']

        # Connection generation
        continue_streaming = True
        connection_list = []
        stream_end_index = 0
        stream_start_index = 0
        while continue_streaming:
            stream_start_index = stream_end_index
            for i in range(0, traffic_matrix.shape[0]):
                for j in range(0, traffic_matrix.shape[1]):
                    requested_bit_rate = traffic_matrix[i][j]
                    traffic_matrix_index = [i, j]
                    if requested_bit_rate != 0:
                        connection_list.append(Connection(node_list[i], node_list[j], input_signal_power,
                                                          traffic_matrix_index))
                        stream_end_index = stream_end_index + 1

            # Stream call
            network.stream(connection_list[stream_start_index:stream_end_index], snr_or_latency_choice)

            no_element_changed = True
            traffic_matrix_is_zero = True

            for i in range(0, traffic_matrix.shape[0]):
                for j in range(0, traffic_matrix.shape[1]):
                    for k in range(stream_start_index, stream_end_index):
                        if [i, j] == connection_list[k].traffic_matrix_index:
                            traffic_matrix[i, j] = traffic_matrix[i, j] - connection_list[k].bit_rate
                            if traffic_matrix[i, j] < 0:
                                traffic_matrix[i, j] = 0
                            if connection_list[k].bit_rate != 0:
                                no_element_changed = False
                    if traffic_matrix[i, j] != 0:
                        traffic_matrix_is_zero = False

            if no_element_changed or traffic_matrix_is_zero:
                continue_streaming = False

        snr_list = list()
        latency_list = list()
        bit_rate_list = list()

        # Result printing
        for i in range(len(connection_list)):
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
        network2 = Network(root / json_path2)

        # Fills "successive" attributes of Nodes and Lines
        network2.connect()

        # Fills weighted paths and initialize route_space attributes
        network2.initialize(root / json_path2)

        traffic_matrix2 = generate_traffic_matrix(input_node, output_node, M)

        # Connection generation
        continue_streaming = True
        connection_list2 = []
        stream_end_index = 0
        stream_start_index = 0
        while continue_streaming:
            stream_start_index = stream_end_index
            for i in range(0, traffic_matrix2.shape[0]):
                for j in range(0, traffic_matrix2.shape[1]):
                    requested_bit_rate = traffic_matrix2[i][j]
                    traffic_matrix_index = [i, j]
                    if requested_bit_rate != 0:
                        connection_list2.append(Connection(node_list[i], node_list[j], input_signal_power,
                                                           traffic_matrix_index))
                        stream_end_index = stream_end_index + 1

            # Stream call
            network2.stream(connection_list2[stream_start_index:stream_end_index], snr_or_latency_choice)

            no_element_changed = True
            traffic_matrix_is_zero = True

            for i in range(0, traffic_matrix2.shape[0]):
                for j in range(0, traffic_matrix2.shape[1]):
                    for k in range(len(connection_list2)):
                        if [i, j] == connection_list2[k].traffic_matrix_index:
                            traffic_matrix2[i, j] = traffic_matrix2[i, j] - connection_list2[k].bit_rate
                            if traffic_matrix2[i, j] < 0:
                                traffic_matrix2[i, j] = 0
                            if connection_list2[k].bit_rate != 0:
                                no_element_changed = False
                    if traffic_matrix2[i, j] != 0:
                        traffic_matrix_is_zero = False

            if no_element_changed or traffic_matrix_is_zero:
                continue_streaming = False

        snr_list2 = list()
        latency_list2 = list()
        bit_rate_list2 = list()

        # Result printing
        for i in range(len(connection_list2)):
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
        network3 = Network(root / json_path3)

        # Fills "successive" attributes of Nodes and Lines
        network3.connect()

        # Fills weighted paths and initialize route_space attributes
        network3.initialize(root / json_path3)

        traffic_matrix3 = generate_traffic_matrix(input_node, output_node, M)

        # Connection generation
        continue_streaming = True
        connection_list3 = []
        stream_end_index = 0
        stream_start_index = 0
        while continue_streaming:
            stream_start_index = stream_end_index
            for i in range(0, traffic_matrix3.shape[0]):
                for j in range(0, traffic_matrix3.shape[1]):
                    requested_bit_rate = traffic_matrix3[i][j]
                    traffic_matrix_index = [i, j]
                    if requested_bit_rate != 0:
                        connection_list3.append(Connection(node_list[i], node_list[j], input_signal_power,
                                                           traffic_matrix_index))
                        stream_end_index = stream_end_index + 1

            # Stream call
            network3.stream(connection_list3[stream_start_index:stream_end_index], snr_or_latency_choice)

            no_element_changed = True
            traffic_matrix_is_zero = True

            for i in range(0, traffic_matrix3.shape[0]):
                for j in range(0, traffic_matrix3.shape[1]):
                    for k in range(len(connection_list3)):
                        if [i, j] == connection_list3[k].traffic_matrix_index:
                            traffic_matrix3[i, j] = traffic_matrix3[i, j] - connection_list3[k].bit_rate
                            if traffic_matrix3[i, j] < 0:
                                traffic_matrix3[i, j] = 0
                            if connection_list3[k].bit_rate != 0:
                                no_element_changed = False
                    if traffic_matrix3[i, j] != 0:
                        traffic_matrix_is_zero = False

            if no_element_changed or traffic_matrix_is_zero:
                continue_streaming = False

        snr_list3 = list()
        latency_list3 = list()
        bit_rate_list3 = list()

        # Result printing
        for i in range(len(connection_list3)):
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

        file_path = 'Results/Lab10/bit_rates_and_capacities_M_' + str(M) + '.txt'
        file = open(root / file_path, "w")
        file.write('M = ' + str(M) + '\n\n')
        bit_rate_mean = np.mean(bit_rate_array)
        bit_rate_mean2 = np.mean(bit_rate_array2)
        bit_rate_mean3 = np.mean(bit_rate_array3)
        print('Bit rate mean for full fixed rate = ', bit_rate_mean, 'Gbps')
        file.write('Bit rate mean for full fixed rate = ' + str(bit_rate_mean) + ' Gbps\n')
        print('Bit rate mean for full flex rate = ', bit_rate_mean2, 'Gbps')
        file.write('Bit rate mean for full flex rate = ' + str(bit_rate_mean2) + ' Gbps\n')
        print('Bit rate mean for full shannon = ', bit_rate_mean3, 'Gbps')
        file.write('Bit rate mean for full shannon = ' + str(bit_rate_mean3) + ' Gbps\n\n')

        capacity = np.sum(bit_rate_array)
        capacity2 = np.sum(bit_rate_array2)
        capacity3 = np.sum(bit_rate_array3)
        print('Capacity for full fixed rate = ', capacity, 'Gbps')
        file.write('Capacity for full fixed rate = ' + str(capacity) + ' Gbps\n')
        print('Capacity mean for full flex rate = ', capacity2, 'Gbps')
        file.write('Capacity for full flex rate = ' + str(capacity2) + ' Gbps\n')
        print('Capacity mean for full shannon = ', capacity3, 'Gbps')
        file.write('Capacity for full shannon = ' + str(capacity3) + ' Gbps\n')
        file.close()

        # Result plotting

        fig_path1 = 'Results/Lab10/snr_distribution_fixed_M_' + str(M) + '.png'
        fig_path2 = 'Results/Lab10/latency_distribution_fixed_M_' + str(M) + '.png'
        fig_path3 = 'Results/Lab10/bit_rate_distribution_fixed_M_' + str(M) + '.png'
        plt.hist(snr_array, color='blue', edgecolor='black', bins=50)
        plt.savefig(root / fig_path1)
        plt.show()
        plt.hist(latency_array, color='blue', edgecolor='black', bins=50)
        plt.savefig(root / fig_path2)
        plt.show()
        plt.hist(bit_rate_array, color='blue', edgecolor='black', bins=50)
        plt.savefig(root / fig_path3)
        plt.show()

        fig_path4 = 'Results/Lab10/snr_distribution_flex_M_' + str(M) + '.png'
        fig_path5 = 'Results/Lab10/latency_distribution_flex_M_' + str(M) + '.png'
        fig_path6 = 'Results/Lab10/bit_rate_distribution_flex_M_' + str(M) + '.png'
        plt.hist(snr_array2, color='blue', edgecolor='black', bins=50)
        plt.savefig(root / fig_path4)
        plt.show()
        plt.hist(latency_array2, color='blue', edgecolor='black', bins=50)
        plt.savefig(root / fig_path5)
        plt.show()
        plt.hist(bit_rate_array2, color='blue', edgecolor='black', bins=50)
        plt.savefig(root / fig_path6)
        plt.show()

        fig_path7 = 'Results/Lab10/snr_distribution_shannon_M_' + str(M) + '.png'
        fig_path8 = 'Results/Lab10/latency_distribution_shannon_M_' + str(M) + '.png'
        fig_path9 = 'Results/Lab10/bit_rate_distribution_shannon_M_' + str(M) + '.png'
        plt.hist(snr_array3, color='blue', edgecolor='black', bins=50)
        plt.savefig(root / fig_path7)
        plt.show()
        plt.hist(latency_array3, color='blue', edgecolor='black', bins=50)
        plt.savefig(root / fig_path8)
        plt.show()
        plt.hist(bit_rate_array3, color='blue', edgecolor='black', bins=50)
        plt.savefig(root / fig_path9)
        plt.show()


if __name__ == '__main__':
    main()
