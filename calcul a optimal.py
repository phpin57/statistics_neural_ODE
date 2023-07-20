import math

def calculate_optimal_range():
    precision = 0.0001
    a = math.sqrt(2)
    b = math.sqrt(2)
    a_lower = 0.0
    b_lower = 0.0

    while a - a_lower > precision and b - b_lower > precision:
        a_mid = (a + a_lower) / 2.0
        b_mid = (b + b_lower) / 2.0

        expression = ((1 + a_mid) * (1 - (b_mid**2 / 2)) - 1 - a_mid * math.sqrt(a_mid**2 + b_mid**2)) * ((1 + a_mid) * (1 - (b_mid**2 / 2)) - 1 + a_mid * math.sqrt(a_mid**2 + b_mid**2))

        if expression > 0:
            a_lower = a_mid
            b_lower = b_mid
        else:
            a = a_mid
            b = b_mid

    return a_lower, b_lower

# Call the function to calculate the optimal range
a_optimal, b_optimal = calculate_optimal_range()

# Print the results
print("Optimal range for a:", a_optimal)
print("Optimal range for b:", b_optimal)
