# Quantitative Factor Model Development

YOUNG_0622_0822

This project focuses on developing and analyzing quantitative factor models for financial markets. It includes initial block attempts ,automated factor processing systems, and market regime analysis using GMM-HMM-KMeans methods.

## Table of Contents

- [Quantitative Factor Model Development](#quantitative-factor-model-development)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [Automated Factor Process System](#automated-factor-process-system)
    - [0702\_0705 Analysis](#0702_0705-analysis)
    - [0715\_0719 Analysis](#0715_0719-analysis)
    - [0722\_0727 Analysis](#0722_0727-analysis)
    - [0729\_0803 Analysis](#0729_0803-analysis)
    - [0805\_0811 Analysis](#0805_0811-analysis)
    - [0812\_0818 Analysis](#0812_0818-analysis)
    - [0819\_0825 Analysis](#0819_0825-analysis)
  - [GMM-HMM-KMeans Market Regimes](#gmm-hmm-kmeans-market-regimes)
  - [Initial Block Attempt](#initial-block-attempt)

## Project Structure

```
├── Automated Factor Process System_0702_0705/
├── Automated Factor Process System_0715_0719/
├── Automated Factor Process System_0722_0727/
├── Automated Factor Process System_0729_0803/
├── Automated Factor Process System_0805_0811/
├── Automated Factor Process System_0812_0818/
├── Automated Factor Process System_0819_0825/
├── file/
│   ├── Factor(alpha) decay framework.pdf
│   ├── Model Construction Outcome.pdf
│   ├── Single Factor Models Fitting, SMF.pdf
├── gmm_hmm_kmeans_market_regimes_0702_0705/
├── gmm_hmm_kmeans_market_regimes_0805_0811/
├── Initial block attempt_0625_0630/
└── README.md
```

## Installation

To set up this project, follow these steps:

1. Clone the repository:
   ```
   git clone https://gitlab.com/ernest_yuen/qfm_dev.git
   ```
2. Install required dependencies (list any specific dependencies here)
3. Set up any necessary environment variables or configuration files

## Automated Factor Process System

Our Automated Factor Process System conducts comprehensive analyses across different date ranges, each focusing on specific aspects of quantitative factor modeling and financial analysis.

### 0702_0705 Analysis
This analysis focuses on the foundational elements of our factor modeling process:
- **Linear Model Setup**: Establishing the basic framework for linear factor models, including model specifications and initial parameter estimations.
- **Single Factor Models Fitting (SMF)**: Detailed process of fitting individual factor models, assessing their performance, and analyzing their predictive power in isolation.

### 0715_0719 Analysis
This period saw significant advancements in our data processing and factor discovery capabilities:
- **Factor Computation Pipeline**: Development of an efficient, scalable pipeline for calculating a wide range of financial factors from raw market data.
- **Automated Crypto Factor Mining System**: Implementation of a system specifically designed to discover and evaluate factors in the cryptocurrency markets, leveraging unique characteristics of crypto assets.

### 0722_0727 Analysis
This phase expanded our factor library and introduced advanced modeling techniques:
- **WorldQuant 101 & Guotaijunan 191 Factors**: Integration and analysis of two prominent factor sets, providing a comprehensive base for our factor models.
- **Tree Models Development**: Exploration of various tree-based and ensemble methods for factor modeling, including:
  - Linear Regression
  - Lasso Regression
  - Ridge Regression
  - Logistic Regression
  - Linear Discriminant Analysis (LDA)
  - Quadratic Discriminant Analysis (QDA)
  - Decision Trees (Regression and Classification)
  - Random Forest (Regression and Classification)
  - XGBoost (Regression and Classification)
  - AdaBoost (Regression and Classification)
  - Gradient Boosting Machines (GBM) (Regression and Classification)
- **Feature Engineering Tools**: Utilization of advanced libraries like `tsfresh` and `featuretools` to automate and enhance our feature engineering process, enabling the discovery of complex, non-linear factors.

### 0729_0803 Analysis
This period was dedicated to comprehensive performance evaluation:
- **All Backtest Results / Reports**: Compilation and analysis of backtesting results for all developed models and factors. This includes performance metrics, risk assessments, and comparative analyses across different market conditions.

### 0805_0811 Analysis
Focus on robustness and generalization:
- **Various Dataset Tests**: Rigorous testing of our models and factors across multiple datasets, including different asset classes, time periods, and market conditions. This phase aimed to ensure the robustness and generalizability of our approaches.

### 0812_0818 Analysis
Final refinement and documentation:
- **Process Modifications**: Implementing final adjustments to our factor processing and modeling pipelines based on insights from previous analyses.
- **Results Compilation**: Comprehensive aggregation and interpretation of all results, preparing final reports and insights for stakeholders.

Each of these analytical phases contributes to the overall development of our quantitative factor models, ensuring a thorough, data-driven approach to financial market analysis.

### 0819_0825 Analysis
- **Recent Month Results**: Process and analyze the most recent market data (July-August), Update factor models with the latest information to generate new results

## GMM-HMM-KMeans Market Regimes

This section focuses on market regime analysis using Gaussian Mixture Models (GMM), Hidden Markov Models (HMM), and K-Means clustering. 


## Initial Block Attempt

The Initial Block Attempt (0625_0630) represents our first approach to Automated Factor Process System.
