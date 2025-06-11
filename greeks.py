import numpy as np
from scipy.stats import norm

def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    elif option_type == 'put':
        delta = norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    else:
        raise ValueError("Only choose 'call' or 'put'")

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}

if __name__ == '__main__':
    # Example parameters
    S = 100      # Spot price
    K = 100      # Strike price
    T = 1        # Time to maturity (1 year)
    r = 0.05     # Risk-free interest rate (5%)
    sigma = 0.2  # Volatility (20%)

    call_greeks = black_scholes_greeks(S, K, T, r, sigma, option_type='call')
    put_greeks = black_scholes_greeks(S, K, T, r, sigma, option_type='put')

    print("Call Option Greeks:")
    for greek, value in call_greeks.items():
        print(f"  {greek.capitalize()}: {value:.6f}")

    print("\nPut Option Greeks:")
    for greek, value in put_greeks.items():
        print(f"  {greek.capitalize()}: {value:.6f}")
