import numpy as np
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculates the Black-Scholes option price.

    Args:
        S (float): Current stock price.
        K (float): Strike price.
        T (float): Time to maturity (in years).
        r (float): Risk-free interest rate (annualized).
        sigma (float): Volatility of the underlying asset (annualized).
        option_type (str): 'call' or 'put'.

    Returns:
        float: Option price.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type == 'put':
        price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    else:
        raise ValueError("Invalid option type. Choose 'call' or 'put'.")
    return price

if __name__ == '__main__':
    # Example usage:
    S = 100  # Current stock price
    K = 100  # Strike price
    T = 1    # Time to maturity (1 year)
    r = 0.05 # Risk-free interest rate (5%)
    sigma = 0.2 # Volatility (20%)

    call_price = black_scholes(S, K, T, r, sigma, option_type='call')
    put_price = black_scholes(S, K, T, r, sigma, option_type='put')

    print(f"The Black-Scholes call option price is: {call_price:.2f}")
    print(f"The Black-Scholes put option price is: {put_price:.2f}")
