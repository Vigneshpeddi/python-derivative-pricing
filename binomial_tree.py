import numpy as np

'''
    Arguments:
        S0 (float): Initial stock price
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate (annualized)
        sigma (float): Volatility of the underlying asset (annualized)
        N (int): Number of steps in the binomial tree
        option_type (str): 'call' or 'put'.

    Returns:
        float: Option price.
'''
def binomial_tree_option_pricing(S0, K, T, r, sigma, N, option_type='call'):
    dt = T / N  
    u = np.exp(sigma * np.sqrt(dt))  
    d = 1 / u  
    p = (np.exp(r * dt) - d) / (u - d)  

    asset_prices = np.zeros(N + 1)
    for i in range(N + 1):
        asset_prices[i] = S0 * (u ** (N - i)) * (d ** i)

    option_values = np.zeros(N + 1)
    if option_type == 'call':
        option_values = np.maximum(0, asset_prices - K)
    elif option_type == 'put':
        option_values = np.maximum(0, K - asset_prices)

    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option_values[j] = np.exp(-r * dt) * (p * option_values[j] + (1 - p) * option_values[j + 1])

    return option_values[0]

if __name__ == '__main__':
    # call the function here
