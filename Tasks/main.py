import random as rnd
import numpy as np
import matplotlib.pyplot as plt

from Core.elements import Network
from Core.utils import json_path1
from Core.utils import json_path2
from Core.utils import json_path3
from Core.utils import number_of_connections
from Core.utils import snr_or_latency_choice

from pathlib import Path

root = Path(__file__).parent.parent


def main():
    for M in range(1, 11):

        # Initialize an object network of class Network
        network = Network(root / json_path1)

        # Fills "successive" attributes of Nodes and Ligines
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

        traffic_matrix = network.generate_traffic_matrix(input_node, output_node, M)
        node_list = ['A', 'B', 'C', 'D', 'E', 'F']

        # writes the input traffic matrix to the output file
        traffic_matrix_mat = np.asmatrix(traffic_matrix)
        traffic_matrix_1_path = 'Results/Lab10/traffic_matrix_fixed_M_' + str(M) + '.txt'
        with open(root / traffic_matrix_1_path, "w") as out_file_1:
            for line in traffic_matrix_mat:
                np.savetxt(out_file_1, line, fmt='%.2f')
        file = open(root / traffic_matrix_1_path, "a")
        file.write('\n')
        file.close()

        # Connection generation
        connections_management_output = network.connections_management(traffic_matrix, node_list, snr_or_latency_choice)
        traffic_matrix = connections_management_output[0]
        connection_list = connections_management_output[1]

        # writes the output matrix to the output file
        traffic_matrix_mat = np.asmatrix(traffic_matrix)
        traffic_matrix_mat = network.handle_output_traffix_matrix(traffic_matrix_mat)
        with open(root / traffic_matrix_1_path, "a") as out_file_1:
            for line in traffic_matrix_mat:
                np.savetxt(out_file_1, line, fmt='%.2f')

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

        traffic_matrix2 = network2.generate_traffic_matrix(input_node, output_node, M)

        # writes the input traffic matrix to the output file
        traffic_matrix_mat2 = np.asmatrix(traffic_matrix2)
        traffic_matrix_2_path = 'Results/Lab10/traffic_matrix_flex_M_' + str(M) + '.txt'
        with open(root / traffic_matrix_2_path, "w") as out_file_2:
            for line in traffic_matrix_mat2:
                np.savetxt(out_file_2, line, fmt='%.2f')
        file = open(root / traffic_matrix_2_path, "a")
        file.write('\n')
        file.close()

        # Connection generation
        connections_management_output2 = network2.connections_management(traffic_matrix2, node_list, snr_or_latency_choice)
        traffic_matrix2 = connections_management_output2[0]
        connection_list2 = connections_management_output2[1]

        # writes the output matrix to the output file
        traffic_matrix_mat2 = np.asmatrix(traffic_matrix2)
        traffic_matrix_mat2 = network2.handle_output_traffix_matrix(traffic_matrix_mat2)
        with open(root / traffic_matrix_2_path, "a") as out_file_2:
            for line in traffic_matrix_mat2:
                np.savetxt(out_file_2, line, fmt='%.2f')

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

        traffic_matrix3 = network3.generate_traffic_matrix(input_node, output_node, M)

        # writes the input traffic matrix to the output file
        traffic_matrix_mat3 = np.asmatrix(traffic_matrix3)
        traffic_matrix_3_path = 'Results/Lab10/traffic_matrix_shannon_M_' + str(M) + '.txt'
        with open(root / traffic_matrix_3_path, "w") as out_file_3:
            for line in traffic_matrix_mat3:
                np.savetxt(out_file_3, line, fmt='%.2f')
        file = open(root / traffic_matrix_3_path, "a")
        file.write('\n')
        file.close()

        # Connection generation
        connections_management_output3 = network3.connections_management(traffic_matrix3, node_list, snr_or_latency_choice)
        traffic_matrix3 = connections_management_output3[0]
        connection_list3 = connections_management_output3[1]

        # writes the output matrix to the output file
        traffic_matrix_mat3 = np.asmatrix(traffic_matrix3)
        traffic_matrix_mat3 = network3.handle_output_traffix_matrix(traffic_matrix_mat3)
        with open(root / traffic_matrix_3_path, "a") as out_file_3:
            for line in traffic_matrix_mat3:
                np.savetxt(out_file_3, line, fmt='%.2f')

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

        file_path = 'Results/Lab10/bit_rates_capacities_and_blocking_events_M_' + str(M) + '.txt'
        file = open(root / file_path, "w")

        file.write('M = ' + str(M) + '\n\n')

        file.write('- Blocking events:\n')
        print('Blocking events for full fixed rate = ', number_of_blocks_full_fixed)
        file.write('\tBlocking events (Fixed-Rate) = ' + str(number_of_blocks_full_fixed) + '\n')
        print('Blocking events for full flex rate = ', number_of_blocks_full_flex)
        file.write('\tBlocking events (Flex-Rate) = ' + str(number_of_blocks_full_flex) + '\n')
        print('Blocking events for full shannon = ', number_of_blocks_full_shannon)
        file.write('\tBlocking events (Shannon) = ' + str(number_of_blocks_full_shannon) + '\n\n')

        file.write('- Bit Rate mean:\n')
        bit_rate_mean = np.mean(bit_rate_array)
        bit_rate_mean2 = np.mean(bit_rate_array2)
        bit_rate_mean3 = np.mean(bit_rate_array3)
        print('Bit rate mean for full fixed rate = ', bit_rate_mean, 'Gbps')
        file.write('\tBit rate mean for full fixed rate = ' + str(bit_rate_mean) + ' Gbps\n')
        print('Bit rate mean for full flex rate = ', bit_rate_mean2, 'Gbps')
        file.write('\tBit rate mean for full flex rate = ' + str(bit_rate_mean2) + ' Gbps\n')
        print('Bit rate mean for full shannon = ', bit_rate_mean3, 'Gbps')
        file.write('\tBit rate mean for full shannon = ' + str(bit_rate_mean3) + ' Gbps\n\n')

        file.write('- Capacities:\n')
        capacity = np.sum(bit_rate_array)
        capacity2 = np.sum(bit_rate_array2)
        capacity3 = np.sum(bit_rate_array3)
        print('Capacity for full fixed rate = ', capacity, 'Gbps')
        file.write('\tCapacity for full fixed rate = ' + str(capacity) + ' Gbps\n')
        print('Capacity mean for full flex rate = ', capacity2, 'Gbps')
        file.write('\tCapacity for full flex rate = ' + str(capacity2) + ' Gbps\n')
        print('Capacity mean for full shannon = ', capacity3, 'Gbps')
        file.write('\tCapacity for full shannon = ' + str(capacity3) + ' Gbps\n')
        file.close()

        # Result plotting

        fig_path1 = 'Results/Lab10/snr_distribution_fixed_M_' + str(M) + '.png'
        fig_path2 = 'Results/Lab10/latency_distribution_fixed_M_' + str(M) + '.png'
        fig_path3 = 'Results/Lab10/bit_rate_distribution_fixed_M_' + str(M) + '.png'
        plt.hist(snr_array, color='blue', edgecolor='black', bins=50)
        plt.xlabel('SNR [dB]')
        plt.ylabel('Frequency')
        plt.title('SNR Distribution (Fixed-Rate, M = ' + str(M) + ')')
        plt.savefig(root / fig_path1)
        plt.show()
        plt.hist(latency_array * 1e3, color='red', edgecolor='black', bins=50)
        plt.xlabel('Latency [ms]')
        plt.ylabel('Frequency')
        plt.title('Latency Distribution (Fixed-Rate, M = ' + str(M) + ')')
        plt.savefig(root / fig_path2)
        plt.show()
        plt.hist(bit_rate_array, color='green', edgecolor='black', bins=50)
        plt.xlabel('Bit Rate [Gbps]')
        plt.ylabel('Frequency')
        plt.title('Bit Rate Distribution (Fixed-Rate, M = ' + str(M) + ')')
        plt.savefig(root / fig_path3)
        plt.show()

        fig_path4 = 'Results/Lab10/snr_distribution_flex_M_' + str(M) + '.png'
        fig_path5 = 'Results/Lab10/latency_distribution_flex_M_' + str(M) + '.png'
        fig_path6 = 'Results/Lab10/bit_rate_distribution_flex_M_' + str(M) + '.png'
        plt.hist(snr_array2, color='blue', edgecolor='black', bins=50)
        plt.xlabel('SNR [dB]')
        plt.ylabel('Frequency')
        plt.title('SNR Distribution (Flex-Rate, M = ' + str(M) + ')')
        plt.savefig(root / fig_path4)
        plt.show()
        plt.hist(latency_array2 * 1e3, color='red', edgecolor='black', bins=50)
        plt.xlabel('Latency [ms]')
        plt.ylabel('Frequency')
        plt.title('Latency Distribution (Flex-Rate, M = ' + str(M) + ')')
        plt.savefig(root / fig_path5)
        plt.show()
        plt.hist(bit_rate_array2, color='green', edgecolor='black', bins=50)
        plt.xlabel('Bit Rate [Gbps]')
        plt.ylabel('Frequency')
        plt.title('Bit Rate Distribution (Flex-Rate, M = ' + str(M) + ')')
        plt.savefig(root / fig_path6)
        plt.show()

        fig_path7 = 'Results/Lab10/snr_distribution_shannon_M_' + str(M) + '.png'
        fig_path8 = 'Results/Lab10/latency_distribution_shannon_M_' + str(M) + '.png'
        fig_path9 = 'Results/Lab10/bit_rate_distribution_shannon_M_' + str(M) + '.png'
        plt.hist(snr_array3, color='blue', edgecolor='black', bins=50)
        plt.xlabel('SNR [dB]')
        plt.ylabel('Frequency')
        plt.title('SNR Distribution (Shannon, M = ' + str(M) + ')')
        plt.savefig(root / fig_path7)
        plt.show()
        plt.hist(latency_array3 * 1e3, color='red', edgecolor='black', bins=50)
        plt.xlabel('Latency [ms]')
        plt.ylabel('Frequency')
        plt.title('Latency Distribution (Shannon, M = ' + str(M) + ')')
        plt.savefig(root / fig_path8)
        plt.show()
        plt.hist(bit_rate_array3, color='green', edgecolor='black', bins=50)
        plt.xlabel('Bit Rate [Gbps]')
        plt.ylabel('Frequency')
        plt.title('Bit Rate Distribution (Shannon, M = ' + str(M) + ')')
        plt.savefig(root / fig_path9)
        plt.show()


if __name__ == '__main__':
    main()
