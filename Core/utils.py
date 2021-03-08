import random as rnd
import numpy as np
import matplotlib.pyplot as plt

c = 3 * (10 ** 8)                # Speed of light
c_network = 2/3 * c              # Speed of signal on the network
json_path1 = 'Resources/nodes_full_fixed_rate.json'  # Json file address
json_path2 = 'Resources/nodes_full_flex_rate.json'  # Json file address
json_path3 = 'Resources/nodes_full_shannon.json'  # Json file address
snr_or_latency_choice = 'snr'
# snr_or_latency_choice = 'latency'
number_of_connections = 100
input_signal_power = 0.001  # [W]

Rs = 32000000  # Symbol rate
Bn = 12500000  # Noise bandwidth
BERt = 10 ** (-3)  # Bit Error Rate


def generate_requests():
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
    return [input_node, output_node]


def generate_output_arrays(connection_list):
    snr_list = list()
    latency_list = list()
    bit_rate_list = list()

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
    return [snr_array, latency_array, bit_rate_array, number_of_blocks_full_fixed]


def plot_results(root, number_of_blocks_full_fixed, number_of_blocks_full_flex, number_of_blocks_full_shannon,
                 bit_rate_array, bit_rate_array2, bit_rate_array3, snr_array, snr_array2, snr_array3, latency_array,
                 latency_array2, latency_array3):
    file = open(root / 'Results/Lab8/bit_rates_capacities_and_blocking_events.txt', "w")

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

    file.write('- Capacity:\n')
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

    plt.hist(snr_array, color='blue', edgecolor='black', bins=50)
    plt.xlabel('SNR [dB]')
    plt.ylabel('Frequency')
    plt.title('SNR Distribution (Fixed-Rate)')
    plt.savefig(root / 'Results/Lab8/snr_distribution_fixed.png')
    plt.show()
    plt.hist(latency_array*1e3, color='red', edgecolor='black', bins=50)
    plt.xlabel('Latency [ms]')
    plt.ylabel('Frequency')
    plt.title('Latency Distribution (Fixed-Rate)')
    plt.savefig(root / 'Results/Lab8/latency_distribution_fixed.png')
    plt.show()
    plt.hist(bit_rate_array, color='green', edgecolor='black', bins=50)
    plt.xlabel('Bit Rate [Gbps]')
    plt.ylabel('Frequency')
    plt.title('Bit Rate Distribution (Fixed-Rate)')
    plt.savefig(root / 'Results/Lab8/bit_rate_distribution_fixed.png')
    plt.show()

    plt.hist(snr_array2, color='blue', edgecolor='black', bins=50)
    plt.xlabel('SNR [dB]')
    plt.ylabel('Frequency')
    plt.title('SNR Distribution (Flex-Rate)')
    plt.savefig(root / 'Results/Lab8/snr_distribution_flex.png')
    plt.show()
    plt.hist(latency_array2*1e3, color='red', edgecolor='black', bins=50)
    plt.xlabel('Latency [ms]')
    plt.ylabel('Frequency')
    plt.title('Latency Distribution (Flex-Rate)')
    plt.savefig(root / 'Results/Lab8/latency_distribution_flex.png')
    plt.show()
    plt.hist(bit_rate_array2, color='green', edgecolor='black', bins=50)
    plt.xlabel('Bit Rate [Gbps]')
    plt.ylabel('Frequency')
    plt.title('Bit Rate Distribution (Flex-Rate)')
    plt.savefig(root / 'Results/Lab8/bit_rate_distribution_flex.png')
    plt.show()

    plt.hist(snr_array3, color='blue', edgecolor='black', bins=50)
    plt.xlabel('SNR [dB]')
    plt.ylabel('Frequency')
    plt.title('SNR Distribution (Shannon)')
    plt.savefig(root / 'Results/Lab8/snr_distribution_shannon.png')
    plt.show()
    plt.hist(latency_array3*1e3, color='red', edgecolor='black', bins=50)
    plt.xlabel('Latency [ms]')
    plt.ylabel('Frequency')
    plt.title('Latency Distribution (Shannon)')
    plt.savefig(root / 'Results/Lab8/latency_distribution_shannon.png')
    plt.show()
    plt.hist(bit_rate_array3, color='green', edgecolor='black', bins=50)
    plt.xlabel('Bit Rate [Gbps]')
    plt.ylabel('Frequency')
    plt.title('Bit Rate Distribution (Shannon)')
    plt.savefig(root / 'Results/Lab8/bit_rate_distribution_shannon.png')
    plt.show()
    return
