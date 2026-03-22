def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Write code here
    for i in range(steps):
        derivative = 2*a*x0 + b
        # there is built in function in sympy library for derivative. can do from there also
        x0 -= lr*derivative

    return x0