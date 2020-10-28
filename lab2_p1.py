import pandas as pd
import matplotlib.pyplot as plt


def main():
    ex1()
    ex2()
    ex3()
    ex4()
    ex5()
    ex6()
    ex7()


def ex1():
    df = pd.read_csv('resources_lab2/sales_data.csv')
    profit_list = df['total_profit'].values
    months = df['month_number'].values
    plt.figure(1)
    plt.plot(months, profit_list, label='Month-wise Profit data of last year ')
    plt.xlabel('Month number')
    plt.ylabel('Profit [$]')
    plt.xticks(months)
    plt.title('Company profit per month ')
    plt.yticks([100e3, 200e3, 300e3, 400e3, 500e3])
    plt.show()


def ex2():
    df = pd.read_csv('resources_lab2/sales_data.csv')
    profit_list = df['total_profit'].values
    months = df['month_number'].values
    plt.figure(2)
    plt.plot(months, profit_list, label='Profit data of last year ',
             color='r', marker='o', markerfacecolor='k', linestyle='--', linewidth=3)
    plt.xlabel('Month Number')
    plt.ylabel('Profit in dollar')
    plt.legend(loc='lower right')
    plt.title('Company Sales data of last year')
    plt.xticks(months)
    plt.yticks([100e3, 200e3, 300e3, 400e3, 500e3])
    plt.show()


def ex3():
    df = pd.read_csv('resources_lab2/sales_data.csv')
    months = df['month_number'].values
    face_cream_sales_data = df['facecream'].values
    face_wash_sales_data = df['facewash'].values
    tooth_paste_sales_data = df['toothpaste'].values
    bathing_soap_sales_data = df['bathingsoap'].values
    shampoo_sales_data = df['shampoo'].values
    moisturizer_sales_data = df['moisturizer'].values
    plt.figure(3)
    plt.plot(months, face_cream_sales_data,
             label='Face cream Sales Data', marker='o', linewidth=3)
    plt.plot(months, face_wash_sales_data,
             label='Face wash Sales Data', marker='o', linewidth=3)
    plt.plot(months, tooth_paste_sales_data,
             label='ToothPaste Sales Data', marker='o', linewidth=3)
    plt.plot(months, bathing_soap_sales_data,
             label='Bathing Soap Sales Data', marker='o', linewidth=3)
    plt.plot(months, shampoo_sales_data,
             label='Shampoo Sales Data', marker='o', linewidth=3)
    plt.plot(months, moisturizer_sales_data,
             label='Moisturizer Sales Data', marker='o', linewidth=3)
    plt.xlabel('Month Number ')
    plt.ylabel('Sales units in number')
    plt.legend(loc='upper left')
    plt.xticks(months)
    plt.yticks([1e3, 2e3, 4e3, 6e3, 8e3, 10e3, 12e3, 15e3, 18e3])
    plt.title('Sales data')
    plt.show()


def ex4():
    df = pd.read_csv('resources_lab2/sales_data.csv')
    months = df['month_number'].tolist()
    tooth_paste_sales_data = df['toothpaste'].values
    plt.figure(4)
    plt.scatter(months, tooth_paste_sales_data, label='Tooth paste sales data')
    plt.xlabel('Month Number')
    plt.ylabel('Number of units sold')
    plt.legend(loc='upper left')
    plt.title('Tooth paste sales data')
    plt.xticks(months)
    plt.grid(True, linewidth=0.5, linestyle='--')
    plt.show()


def ex5():
    df = pd.read_csv('resources_lab2/sales_data.csv')
    months = df['month_number'].tolist()
    bathing_soap_sales_data = df['bathingsoap'].tolist()
    plt.figure(5)
    plt.bar(months, bathing_soap_sales_data)
    plt.xlabel('Month Number')
    plt.ylabel('Sales units in number')
    plt.xticks(months)
    plt.grid(True, linewidth=0.5, linestyle="--")
    plt.title('Bathing soap sales data')
    plt.savefig('figures_lab2/sales_data_of_bathing_soap.png', dpi=150)
    plt.show()


def ex6():
    df = pd.read_csv('resources_lab2/sales_data.csv')
    profit_list = df['total_profit'].values
    plt.figure(6)
    profit_range = [150e3, 175e3, 200e3, 225e3, 250e3, 300e3, 350e3]
    plt.hist(profit_list, profit_range, label='Profit data')
    plt.xlabel('profit range [$]')
    plt.ylabel('Actual Profit [$]')
    plt.legend(loc='upper left')
    plt.xticks(profit_range)
    plt.title('Profit data')
    plt.show()


def ex7():
    df = pd.read_csv('resources_lab2/sales_data.csv')
    months = df['month_number'].values
    bathing_soap = df['bathingsoap'].values
    face_wash_sales_data = df['facewash'].values
    f, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(months, bathing_soap, label='Bathing soap Sales Data',
                color='k', marker='o', linewidth=3)
    axs[0].set_title('Sales data of a Bathing soap')
    axs[0].grid(True, linewidth=0.5, linestyle='--')
    axs[0].legend()
    axs[1].plot(months, face_wash_sales_data, label='Face Wash Sales Data',
                color='r', marker='o', linewidth=3)
    axs[1].set_title('Sales data of a face wash')
    axs[1].grid(True, linewidth=0.5, linestyle='--')
    axs[1].legend()
    plt.xticks(months)
    plt.xlabel('Month Number')
    plt.ylabel('Sales units in number')
    plt.show()


if __name__ == '__main__':
    main()


