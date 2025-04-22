
# Quantium Retail Analytics – Trial Store Performance Evaluation

## 🏪 Project Title
**Quantium Task 2: Trial Store Performance Evaluation**

## 📋 Objective
The goal of this task is to evaluate the effectiveness of a new store layout implemented in select trial stores by comparing their performance against matched control stores. The findings will help determine whether the new layout should be rolled out across all stores.

## 📁 Files Used
- `QVI_data.csv`: Main transactional dataset containing chip sales, customer, product, and store details.
- `Quantium task 2.ipynb`: Jupyter Notebook with the full implementation of data processing, control store matching, visualization, and analysis.

## 🧠 Key Concepts & Methods

### ✅ Data Preprocessing
- Cleaned and parsed transaction data.
- Converted `DATE` column to datetime and extracted `MONTH`, `YEAR`, `MONTH_YEAR`.

### ✅ Feature Engineering
- Monthly metrics calculated per store:
  - `TOTAL_SALES`
  - `TOTAL_CUSTOMERS`
  - `TOTAL_TRANSACTIONS`
  - `TRANSACTIONS_PER_CUSTOMER`

### ✅ Control Store Selection
- For each trial store (77, 86, 88), identified best-matched control store using:
  - Pearson correlation
  - Magnitude distance across key metrics

### ✅ Trial Analysis Period
- **Pre-trial period:** July 2018 to March 2019  
- **Trial period:** April 2019 to June 2019

### ✅ Statistical Evaluation
- Compared trial vs control store for:
  - Sales
  - Customer count
  - Transactions per customer
- Measured percentage change and overall impact.

## 📈 Key Results

### 🔹 Trial Store 77 (Control: Store 17)
- **+27.38%** in sales
- **+16.63%** in customers
- ✅ **Recommendation: Roll out layout**

### 🔹 Trial Store 86 (Control: Store 13)
- Slight decline in performance
- ⚠ **Recommendation: Consider adjustments before rollout**

### 🔹 Trial Store 88 (Control: Store 201)
- Minimal change or underperformance
- ❌ **Recommendation: Do not roll out layout yet**

## 📌 Final Recommendation
Proceed with layout rollout to stores with characteristics similar to Store 77. Other stores may require further adjustment or testing based on customer behavior and transaction trends.

## 👤 Author
**[Your Name]**  
Quantium Analytics Virtual Internship Program  
April 2025
