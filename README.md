# Weather Insight Analyzer — DAA Mini Project

Upload any weather CSV → the C++ program finds the covariance matrix using
**sequential**, **row-wise threaded** and **column-wise threaded** matrix multiplication →
the Streamlit app shows insights, a what-if predictor and a speed comparison.

## Files
| File | Role |
|---|---|
| `weather_analyzer.cpp` | C++ program: reads CSV, statistics, covariance, correlation, timing |
| `app.py` | Streamlit web app (what the user sees) |
| `sample_weather.csv` | Small sample: 1 city, 365 days, 5 factors |
| `big_weather.csv` | Big sample: 10 Maharashtra cities, 20,000 hourly records, 50 factors |
| `requirements.txt` | Python libraries needed |
| `.streamlit/config.toml` | Lets file upload work in GitHub Codespaces |
| `.devcontainer/devcontainer.json` | Sets up Codespaces automatically |

## Run on your laptop
```bash
pip install -r requirements.txt
g++ -O2 -std=c++17 -pthread weather_analyzer.cpp -o weather_analyzer
streamlit run app.py
```
Run only the C++ program:
```bash
./weather_analyzer big_weather.csv results
```

## Run on GitHub Codespaces
Repository page → **Code** → **Codespaces** → **Create codespace on main**.
Setup runs automatically (installs g++, Python libraries, compiles the program). Then:
```bash
streamlit run app.py
```
Click **Open in Browser** on the popup (or open the **Ports** tab → port 8501 → globe icon).

## Algorithm
1. Mean of every column → centered matrix Xc (n × m)
2. Cov = (Xcᵀ × Xc) / (n − 1)  ← matrix multiplication done 3 ways
3. Correlation r = Cov(X,Y) / (σx · σy)
4. Prediction: Y = slope·X + intercept, where slope = Cov(X,Y) / Var(X)

Time complexity of step 2: **O(m² · n)**. Threading divides the m × m result
by rows or by columns, so each thread does about 1/T of the work.

## Expected result
| File | Sequential | Row-wise | Column-wise |
|---|---|---|---|
| sample_weather.csv (365 × 5) | fastest | slower (thread cost) | slower |
| big_weather.csv (20,000 × 50) | ~175 ms | ~100 ms (1.7×) | ~98 ms (1.8×) |

Measured on 2 cores; more cores give bigger speedups. Small data: creating threads
costs more than the work. Big data: work is shared between cores, so threads win.
