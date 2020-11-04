import json


def main():
    ex1()
    ex2()
    ex3()
    ex4()
    ex5()


def ex1():
    print('Exercise 1')
    json_obj = '{ "Name":"David", "Class":"I", "Age":6}'
    python_obj = json.loads(json_obj)
    print('\nJSON data:')
    print(python_obj)
    print('\nName:', python_obj['Name'])
    print('Class:', python_obj['Class'])
    print('Age:', python_obj['Age'])


def ex2():
    print('\n\nExercise 2')
    # a Python object ( dict ):
    python_obj = {
        'name': 'David ',
        'class ': 'I',
        'age ': 6
    }
    print(type(python_obj))
    # convert into JSON :
    j_data = json.dumps(python_obj)
    # result is a JSON string :
    print(j_data)


def ex3():
    print('\n\nExercise 3')
    python_dict = {'name': 'David', 'age': 6, 'class': 'I'}
    python_list = ['Red', 'Green', 'Black']
    python_str = 'Python Json'
    python_int = 1234
    python_float = 21.34
    python_t = True
    python_f = False
    python_n = None
    json_dict = json.dumps(python_dict)
    json_list = json.dumps(python_list)
    json_str = json.dumps(python_str)
    json_num1 = json.dumps(python_int)
    json_num2 = json.dumps(python_float)
    json_t = json.dumps(python_t)
    json_f = json.dumps(python_f)
    json_n = json.dumps(python_n)
    print('json dict :', json_dict)
    print('json list :', json_list)
    print('json string :', json_str)
    print('json number1 :', json_num1)
    print('json number2 :', json_num2)
    print('json true :', json_t)
    print('json false :', json_f)
    print('json null :', json_n)


def ex4():
    j_str = {'4': 5, '6': 1, '1': 3, '2': 4}
    print('Original String :', j_str)
    print('\nJSON data :')
    print(json.dumps(j_str, sort_keys=True, indent=4))


def ex5():
    with open('Lab2/resources_lab2/states.json') as f:
        state_data = json.load(f)
    print('Original JSON keys: ', [state.keys()
                                   for state in state_data['states']][0])
    for state in state_data['states']:
        del state['area_codes']
    print('\nModified JSON keys: ', [state.keys()
                                     for state in state_data['states']][0])
    with open('Lab2/resources_lab2/new_states.json', 'w') as f:
        json.dump(state_data, f, indent=2)
    with open('Lab2/resources_lab2/new_states.json') as f:
        state_data = json.load(f)
    print('\nReloaded JSON keys: ', [state.keys()
                                     for state in state_data['states']][0])


if __name__ == '__main__':
    main()

