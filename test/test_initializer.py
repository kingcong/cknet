from cknet.initializers import RandomInit

init = RandomInit()
parameters = init.fill([5,4,3])
print(parameters)