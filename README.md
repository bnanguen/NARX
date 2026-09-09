# NARX: Nonlinear AutoRegressive Model with Exogenous Inputs

## Overview

This project explores the implementation and application of a **NARX (Nonlinear AutoRegressive with Exogenous Inputs)** model for time series forecasting.

NARX models extend traditional autoregressive approaches by incorporating both:

- Past values of the target variable (autoregressive component)
- External input variables (exogenous component)

This makes them particularly suitable for modeling complex nonlinear dynamic systems where external factors influence future observations.

---

## Objectives

The main goals of this project are:

- Build a NARX-based forecasting framework
- Process and prepare time series data
- Train and evaluate predictive models
- Compare forecasting performance
- Analyze the impact of exogenous variables on prediction accuracy

---

## Project Structure

```text
NARX/
│
├── notebooks/
│   └── main.ipynb
│
├── data/
│   └── ...
│
├── models/
│   └── ...
│
├── results/
│   └── ...
│
└── README.md

Please make sure to install the requirements in `requirements.txt`

if you are using pip:
```bash
pip install -r requirements.txt
```
a module structure has been chosen to reduce code conflicts and to allow easier collaboration, reusability, and readability.


You can generate the submission using `notebooks/main.ipynb`

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/XVFDvpGt)
