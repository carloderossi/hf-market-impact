# **High‑Frequency Market Impact Model for Monetary Policy Operations**

This repository implements a full simulation and causal‑inference pipeline to study how monetary‑policy operations affect high‑frequency financial markets.  
It is designed as a portfolio‑ready project aligned with the expectations of the **Swiss National Bank (SNB)** for roles involving:

- empirical economics  
- machine learning  
- high‑frequency data  
- monetary‑policy implementation  
- causal inference  

The project includes:

- a realistic synthetic high‑frequency market simulator  
- a DoubleML causal‑inference pipeline  
- optional Causal Forest estimation  
- Azure ML job definitions  
- visualization tools to demonstrate realism of synthetic data  

---

## **1. Project Overview**

Central banks intervene in money and FX markets to implement monetary policy.  
Understanding the *causal impact* of these operations on:

- prices  
- spreads  
- liquidity  
- volatility  

is essential for optimal execution and policy design.

However, real high‑frequency data are often proprietary.  
This project solves that by generating **synthetic but realistic** market data and applying modern causal‑inference methods to recover the effect of simulated policy operations.

---

## **2. Repository Structure**

```
hf-market-impact/
├─ src/
│  └─ hf_impact/
│     ├─ data_generation.py
│     ├─ causal_doubleml.py
│     ├─ causal_forest.py
│     └─ evaluation.py
├─ run_doubleml.py
├─ visualize_market.py
├─ doubleml-job.yml
├─ environment.yml
└─ README.md
```

---

## **3. Synthetic High‑Frequency Market Data**

The simulator produces realistic intraday market behavior by incorporating:

### **Price dynamics**
- A stochastic process with drift and volatility  
- Microstructure noise (bid/ask, spread, jumps)

### **Liquidity & volume**
- Normal vs stressed liquidity regimes  
- Volume clustering around events

### **Policy operations (treatments)**
- Randomly timed interventions  
- Temporary impact on drift, volatility, and spreads

### **Confounders**
- Time‑of‑day  
- Day‑of‑week  
- Synthetic macro sentiment  

This ensures the data exhibit real‑world “stylized facts” such as:

- volatility clustering  
- fat‑tailed returns  
- spread widening during stress  
- event‑driven volume spikes  

---

## **4. Causal Inference Methods**

This project uses two modern causal‑ML approaches:

---

# **CausalForest and DoubleML in plain language**

### **CausalForest**

A causal forest is like a random forest, but instead of predicting a value, it estimates **how a treatment changes an outcome**, possibly differently for different types of observations.

- You have an outcome \(Y\), a treatment \(D\), and covariates \(X\).  
- The forest splits the data into regions where the treatment effect is similar.  
- In each leaf, it compares treated vs untreated observations to estimate a **local treatment effect**.  
- Averaging many trees gives stable estimates of:
  - **ATE** (average treatment effect)  
  - **CATE** (conditional effect depending on market state)  

For this project:  
treatment = policy operation size  
outcome = return or spread change  
covariates = liquidity, volume, spreads, macro sentiment  

It tells you *where* operations have the biggest impact.

---

### **DoubleML**

DoubleML is a framework for causal inference that uses machine learning while avoiding overfitting bias.

It relies on:

- **Neyman orthogonality**  
  → makes the causal estimate insensitive to small ML errors  
- **Sample splitting / cross‑fitting**  
  → prevents overfitting by separating training and estimation  
- **Flexible ML models**  
  → random forests, boosting, neural nets, etc.

The typical model:

\[
Y = \theta D + g(X) + \varepsilon,\quad D = m(X) + v
\]

DoubleML estimates the nuisance functions \(g(X)\) and \(m(X)\) with ML, then isolates the causal effect \(\theta\).

For this project:  
It estimates the causal effect of policy operations on returns or spreads, controlling for market conditions.

---

## **5. Running the Pipeline**

### **Local execution**

Generate data + run DoubleML:

```bash
python run_doubleml.py --n_steps 20000 --output_path results/
```

Visualize synthetic data:

```bash
python visualize_market.py --n_steps 20000
```

---

## **6. Running on Azure ML**

Submit the job:

```bash
az ml job create --file doubleml-job.yml
```

Azure ML will:

- build the environment  
- run the DoubleML pipeline  
- store outputs (CSV, summary, JSON)  

---

## **7. Visualizing Realism**

`visualize_market.py` produces:

- price series  
- return distribution  
- spread dynamics  
- volume clustering  
- operation timestamps  

These plots demonstrate that the synthetic data behave like real high‑frequency markets.

---

## **8. Optional Deep‑Learning Extension**

You can extend the project with:

### **LSTM or Transformer models**
To learn latent market states from:

- price sequences  
- volatility patterns  
- order‑book snapshots  

These embeddings can be fed into DoubleML as additional covariates.

### **NPU acceleration (WinML + ONNX)**
Deep models can be exported to ONNX and accelerated on your Intel NPU.  
Tree‑based causal models remain CPU‑bound.

---
