import yfinance as yf
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime

# %%
# -------------------------
# Black-Scholes Greeks
# -------------------------
def bs_put_greeks(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    put_price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    delta = norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S*sigma*np.sqrt(T))
    theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) +
             r*K*np.exp(-r*T)*norm.cdf(-d2))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    rho = -K*T*np.exp(-r*T)*norm.cdf(-d2)
    
    return put_price, delta, gamma, theta, vega, rho

# -------------------------
# Monte Carlo Simulation
# -------------------------
# def monte_carlo_paths(S0, mu, sigma_annual, days, n_paths):
#     dt = 1/252
#     sigma = sigma_annual / np.sqrt(252)
#     drift = (mu - 0.5*sigma**2) * dt
#     shocks = np.random.normal(size=(days, n_paths))
#     increments = drift + sigma*np.sqrt(dt)*shocks
#     log_paths = np.cumsum(increments, axis=0)
#     paths = S0 * np.exp(log_paths)
#     return paths

def monte_carlo_paths(S0, mu, sigma_annual, days, n_paths):
    dt = 1/252
    drift = (mu - 0.5 * sigma_annual**2) * dt
    shocks = np.random.normal(size=(days, n_paths))
    increments = drift + sigma_annual * np.sqrt(dt) * shocks
    log_paths = np.cumsum(increments, axis=0)
    paths = S0 * np.exp(log_paths)
    return paths


def monte_carlo_breach(paths, strike):
    touch_prob = np.mean((paths <= strike).any(axis=0))
    terminal_prob = np.mean(paths[-1] <= strike)
    return touch_prob, terminal_prob

# -------------------------
# CSP Candidate Finder
# -------------------------
def csp_candidate_strikes(ticker, expiry, max_terminal_prob=0.05, n_paths=10000, risk_free_rate=0.05):
    opt = yf.Ticker(ticker)
    S0 = opt.history(period="1d")['Close'][-1]
    chain = opt.option_chain(expiry)
    puts = chain.puts
    
    # Compute RV from historical data
    hist = opt.history(period="1y")['Close']
    log_returns = np.log(hist / hist.shift(1)).dropna()
    RV_annual = log_returns.std() * np.sqrt(252)
    
    # Time to expiry in years
    today = datetime.today()
    expiry_date = datetime.strptime(expiry, '%Y-%m-%d')
    T = (expiry_date - today).days / 252
    days = (expiry_date - today).days
    
    candidates = []
    
    for idx, row in puts.iterrows():
        strike = row['strike']
        IV = row['impliedVolatility']
        option_price = row['lastPrice']
        
        # Compute Greeks
        put_price_bs, delta, gamma, theta, vega, rho = bs_put_greeks(S0, strike, T, risk_free_rate, IV)
        
        # Monte Carlo with RV
        paths_rv = monte_carlo_paths(S0, mu=0.0, sigma_annual=RV_annual, days=days, n_paths=n_paths)
        touch_rv, terminal_rv = monte_carlo_breach(paths_rv, strike)
        
        if terminal_rv <= max_terminal_prob:
            candidates.append({
                "strike": strike,
                "option_price": option_price,
                "delta": delta,
                "gamma": gamma,
                "theta": theta,
                "vega": vega,
                "rho": rho,
                "IV": IV,
                "RV": RV_annual,
                "touch_prob": touch_rv,
                "terminal_prob": terminal_rv
            })
    
    # Sort by strike descending (OTM first)
    candidates = sorted(candidates, key=lambda x: x["strike"], reverse=True)
    
    return candidates

# %%
# -------------------------
# Example usage
# -------------------------
ticker = "UPRO"
expiry = "2026-05-15"
max_terminal_prob = 0.50  # your risk threshold
candidates = csp_candidate_strikes(ticker, expiry, max_terminal_prob, n_paths=70000, risk_free_rate=0.05)

# Display results
print(f"CSP candidate strikes for {ticker} expiring {expiry} (max terminal prob {max_terminal_prob*100:.1f}%):\n")
for c in candidates:
    print(f"Strike: {c['strike']}, Price: {c['option_price']:.2f}, Δ: {c['delta']:.3f}, gamma: {c['gamma']:.3f}, RV: {c['RV']:.2%}, theta: {c['theta']:.3f} "
          f"TouchProb: {c['touch_prob']:.6%}, TerminalProb: {c['terminal_prob']:.6%}, IV: {c['IV']:.2%}")

# %%
# print("Max simulated move down:", np.min(paths))
# print("Min simulated move up:", np.max(paths))

# %% 

import matplotlib.pyplot as plt

paths = monte_carlo_paths(S0=400, mu=0.0, sigma_annual=0.25, days=35, n_paths=1000)

plt.figure(figsize=(10,6))
plt.plot(paths[:, :20])  # plot first 20 sample paths
plt.title("Monte Carlo Simulated Stock Price Paths")
plt.xlabel("Days")
plt.ylabel("Price")
plt.show()

print(paths.shape)   # e.g. (60, 1000)
print(paths[:5, :3]) # first 5 days of 3 paths
# %%
