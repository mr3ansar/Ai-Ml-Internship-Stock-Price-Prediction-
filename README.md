# 📈 Task 2: Stock Price Prediction (Short-Term)
**DevelopersHub Corporation — AI/ML Engineering Internship**
**Author:** Abdul Samad

---

## 📌 Task Objective
Use 6 years of real Apple stock data to predict the **next trading day's closing price** using two Machine Learning models — then validate against actual live market data.

---

## 📂 Dataset
| Property | Value |
|---|---|
| **Stock** | Apple Inc. (AAPL) |
| **Source** | Yahoo Finance via `yfinance` |
| **Period** | Jan 2020 → Apr 2026 |
| **Trading Days** | 1,561 |
| **Price Range** | $56.09 → $286.19 |
| **Missing Values** | 0 |

---

## 🔍 Data Integrity & Leak Prevention

This notebook went through multiple debugging iterations before producing valid results. Key issues identified and fixed:

### ✅ Fixes Applied
| Issue | How it was fixed |
|---|---|
| yfinance MultiIndex columns | Flattened with `'_'.join(col)` then renamed |
| `auto_adjust=True` conflict | Switched to `auto_adjust=False` to avoid `Adj Close` collision |
| Today's Close used as feature | All features use `.shift(1)` — only yesterday's data |
| Moving averages including today | `data['Close'].shift(1).rolling(n).mean()` — past only |
| Scaler fitted on test data | `fit_transform` on train only, `transform` on test |
| Chronological split violated | No shuffle — `iloc[:split_idx]` / `iloc[split_idx:]` |

### ⚠️ One Remaining Concern (Minor)
The FEATURES list correctly excludes raw `Open`, `High`, `Low`, `Close`, `Volume` columns and uses only the lagged versions (`Prev_Close`, `Prev_High` etc.). However, the raw columns still exist in `data` — they are **not passed to the model**, so there is no actual leakage. This is clean.

### 🧪 Sanity Check Result
```
Today vs Tomorrow correlation: 0.998218  ✅
```
A correlation of 0.998 between today's and tomorrow's price confirms the data is consistent — no hidden splits or scale jumps.

---

## ⚙️ Feature Engineering (15 Features)
All features use only **past data** — no same-day or future information is included.

| Feature | What it captures | Why it helps |
|---|---|---|
| `Prev_Close` | Yesterday's closing price | Strongest single predictor |
| `Prev_High/Low/Open` | Yesterday's full price range | Context for today's expected range |
| `Prev_Volume` | Yesterday's trading activity | High volume = high conviction move |
| `MA5 / MA7` | 5 & 7-day moving average | Short-term price trend direction |
| `MA14 / MA21` | 14 & 21-day moving average | Medium-term trend momentum |
| `Price_Change` | 1-day price shift | Is the stock accelerating or slowing? |
| `Price_Change5` | 5-day price shift | Week-long momentum |
| `Daily_Range` | High minus Low (prev day) | How volatile was yesterday? |
| `Volatility5` | 5-day std deviation | Is the market calm or choppy? |
| `Volume_MA5` | 5-day avg volume | Normal trading level |
| `Volume_Change` | 1-day volume shift | Unusual activity spike detector |

---

## 🤖 Models
| Model | Config |
|---|---|
| **Linear Regression** | Default sklearn — no hyperparameters needed |
| **Random Forest** | 300 trees, max_depth=5, min_samples_leaf=10, max_features=0.7 |

---

## 🏆 Results

### Model Evaluation (Test Set: 313 days, Jan 2025 → Apr 2026)
| Metric | Linear Regression 🏆 | Random Forest |
|---|---|---|
| **MAE** | **$4.48** | $10.77 |
| **RMSE** | **$6.41** | $13.83 |
| **R² Score** | **0.9428 (94.3%)** | 0.7337 (73.4%) |

### Real-World Validation (Live Market Test)
```
Input date:        April 20, 2026
Predicting:        Next trading day's close

Actual close:      $266.17
LR predicted:      $269.91  →  off by $3.74  (1.41% error) ✅
RF predicted:      $248.70  →  off by $17.47 (6.56% error)

Winner: Linear Regression
```

---

## 💡 Key Findings

**1. Linear Regression beat Random Forest — and that's not surprising**
Apple's price doesn't spike randomly. It trends gradually. Linear Regression is built for exactly this kind of smooth, predictable movement. Random Forest adds complexity the data doesn't need.

**2. Yesterday's close is the most powerful feature**
The single best predictor of tomorrow's price is simply what the price was yesterday. All 15 features improve accuracy, but `Prev_Close` does the heaviest lifting.

**3. Moving averages tell the model which direction it's heading**
Without MA5/MA7, the model only knows where Apple was yesterday. With moving averages, it also knows whether the recent trend is up or down — and by how much.

**4. Volatility5 matters most during turbulent periods**
During calm periods it barely changes. But when earnings reports or market events hit, `Volatility5` spikes and warns the model that predictions will be less reliable.

**5. Stock splits silently destroy ML models**
Early attempts on NVDA data produced R² = -12. The 10:1 split in June 2024 made training prices ($10–$140) and test prices ($100–$200) completely incompatible. AAPL's stable price range makes it far better suited for this task.

**6. 94.3% of price movement explained — on a stock it's never seen before**
The model was trained on 2020–2025 data. It was tested on 2025–2026 — a period it never saw. Getting R² = 0.94 on truly unseen data is the real measure of success here.

---

## ⚠️ What R² = 0.94 Actually Means
This does NOT mean the model is ready for real trading. It means it's excellent at predicting the *general level* of tomorrow's price based on smooth historical trends. It will still fail badly during:
- Earnings surprises
- Major news events (product launches, regulatory decisions)
- Macroeconomic shocks

**For educational purposes only.**

---

## 📁 File Structure
```
task2-stock-prediction/
├── stockpred.ipynb          # Main notebook
├── README.md                # This file
└── plots/
    ├── plot1_price_volume.png
    └── actual_vs_predicted.png
```

---

## ▶️ How to Run
```bash
pip install yfinance scikit-learn pandas numpy matplotlib seaborn jupyter
mkdir plots
jupyter notebook stockpred.ipynb
```

---

## 👤 Author
**Abdul Samad**
AI/ML Engineering Intern — DevelopersHub Corporation
