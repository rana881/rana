def tanh(x):
    return (2 / (1 + (2.71828 ** (-2 * x)))) - 1
def dot_product(weights, inputs):
    return sum(w * i for w, i in zip(weights, inputs))
def forward_pass(x, w1, w2, b1, b2):
    z1 = [dot_product(w, x) + b1 for w in w1]
    a1 = [tanh(z) for z in z1]

    z2 = dot_product(w2, a1) + b2
    output = tanh(z2)
    return output
input_size = 3  
hidden_size = 4  
output_size = 1  
w1 = [[-0.2, 0.3, -0.4], [0.1, -0.1, 0.2], [0.4, -0.3, 0.1], [-0.5, 0.2, 0.3]]
w2 = [0.2, -0.3, 0.1, 0.4]
b1 = 0.5
b2 = 0.7
x = [0.1, 0.2, 0.3]
output = forward_pass(x, w1, w2, b1, b2)

print("Output of the network:", output)
