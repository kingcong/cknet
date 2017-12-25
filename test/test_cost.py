from cknet.cost import CrossEntropy
from test.testCases_v3 import compute_cost_test_case

Y, AL = compute_cost_test_case()

cost = CrossEntropy()

print("cost = " + str(cost(AL, Y)))

for index in reversed(range(4)):
    print(index)