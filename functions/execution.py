import os


def save_execution_file(parameters):
    try:
        os.makedirs("./executions/" + parameters['execution_name'] + "/")
        print("Directory ", parameters['execution_name'], " Created ")
    except FileExistsError:
        print("Directory ", parameters['execution_name'], " already exists")
    f = open("./executions/" + parameters['execution_name'] + "/READ_ME.txt", "w")
    f.write("This code was execute with parameters :\n")
    for key, value in parameters.items():
        f.write('* ' + str(key) + ' = \t' + str(value) + '\n')
    f.write('\n')
    f.close()


def add_to_execution_file(parameters, line):
    print(line)
    f = open("./executions/" + str(parameters['execution_name']) + "/READ_ME.txt", "a+")
    f.write(line + '\n')
    f.close()
