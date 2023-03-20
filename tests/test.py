import os


def print_name(name,age):
    print(name, age)
    return 200

#func_name = "print_name"
#resp = globals()[func_name](*["Rama",40])
#print(resp)

os.environ['my_var'] = 'Training'

print(os.environ['my_var'])
