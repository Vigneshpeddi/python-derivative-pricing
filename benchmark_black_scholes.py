import numpy as np
from black_scholes import black_scholes_price
import time

S_range = np.linspace(50, 150, 10)        # Spot price
K_range = np.linspace(50, 150, 10)        # Strike price
r_range = np.linspace(0.01, 0.05, 10)     # Risk-free rate
T_range = np.linspace(0.1, 2.0, 10)       # Time to maturity
sigma_range = np.linspace(0.1, 0.5, 10)   # Volatility

# Track results
total_tests = 0
errors = []

start_time = time.time()

for S in S_range:
    for K in K_range:
        for r in r_range:
            for T in T_range:
                for sigma in sigma_range:
                    try:
                        call_price = black_scholes_price(S, K, r, T, sigma, "call")
                        put_price = black_scholes_price(S, K, r, T, sigma, "put")
                        
                        # Put-call parity check
                        parity_left = call_price - put_price
                        parity_right = S - K * np.exp(-r * T)
                        parity_error = abs(parity_left - parity_right)

                        errors.append(parity_error)
                        total_tests += 2 
                        
                    except Exception as e:
                        print(f"Error with parameters S={S}, K={K}, r={r}, T={T}, sigma={sigma}")
                        print(e)

end_time = time.time()

# Reporting
print(f"\n✅ Total test cases run: {total_tests}")
print(f"⏱️  Total runtime: {end_time - start_time:.2f} seconds")
print(f"📈 Average put-call parity error: {np.mean(errors):.6f}")
print(f"📉 Max error observed: {np.max(errors):.6f}")
print(f"✔️ Error < 0.01 for {np.sum(np.array(errors) < 0.01)} out of {total_tests} cases")
