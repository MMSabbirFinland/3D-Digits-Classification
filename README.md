# 3D Digits Classification

# Project Context
This repository contains a group project completed as part of the **Pattern Recognition and Machine Learning (PRML)** course at **LUT University**.

# Contributors
- **Adrian Khalatbari**
- **Muhammad Monowarul Sabbir**
- **Moriom Akter**

# My Contributions
I contributed to:
- Data exploration and preprocessing  
- Feature extraction pipeline  
- Model training and evaluation  
- Hyperparameter tuning for kNN  
- Documentation and code organization  

This repository is a personal copy of the group project for portfolio and learning purposes.


This project implements a full classification pipeline for recognizing
hand-written digits (0--9) drawn in the air and recorded as 3D motion
data. Each digit is stored as a CSV file containing an **N × 3**
sequence of\
(x, y, z) fingertip positions.

The goal is to implement the required function:

`C = digit_classify(testdata)`

where `testdata` is an **N × 3** motion and `C` is the predicted digit
(0--9).

This README explains **how the project works from start to finish** and\
**how to run every step.**

------------------------------------------------------------------------

# 📁 Project Structure

    STNUM/
    ├── code/
    │    ├── project_steps.py
    │    ├── explore_data.py
    │    ├── extract_features.py
    │    ├── knn.py
    │    ├── train_knn.py
    │    ├── tune_knn.py
    │    ├── train_final_knn.py
    │    ├── digit_classify.py
    │    ├── test_digit_classify.py
    │    └── test_features.py
    └── data/
          ├── stroke_0_0001.csv
          ├── stroke_1_0034.csv
          ├── …

------------------------------------------------------------------------

# 🗂 Project File Overview (Ordered by Project Steps)

| File Name | Purpose / Role                | What It Does | Used During |
|-----------|-------------------------------|--------------|-------------|
| `code/explore_data.py` | Step 1: Explore data          | Loads CSV motions, plots examples, helps understand dataset | Development only |
| `code/extract_features.py` | Step 2: Feature engineering   | Normalizes motions, resamples to fixed length, flattens to feature vector | Training + classification |
| `code/knn.py` | Step 3: Implement classifier  | Manual k-NN classifier (distance + majority vote), no sklearn | Training + classification |
| `code/train_knn.py` | Step 4: Baseline training     | Loads dataset, extracts features, train/val split, tests basic k-NN | Early training |
| `code/tune_knn.py` | Step 5: Hyperparameter tuning | Grid search over `k` and `num_points`, outputs best settings | Model tuning |
| `code/train_final_knn.py` | Step 6: Final training        | Uses best hyperparameters, trains on full data, saves `knn_model.npz` | Final model building |
| `code/digit_classify.py` | Step 7: Final classifier      | Loads saved model, extracts features for one motion, returns predicted digit | Final submission |
| `code/test_digit_classify.py` | Step 8: Test classifier       | Runs `digit_classify` on sample input to verify output | Debugging |
| `code/test_features.py` | Step 9: Feature test          | Tests feature extraction on sample data | Debugging |
| `code/project_steps.py` | Run the project               | Your internal planning file with step documentation | Your organization |
------------------------------------------------------------------------

🚀 **How the System Works (Step-by-Step)**

## **Step 1 --- Data: 3D Motions of Hand-Written Digits**

The `data/` folder contains \1000 CSV files.
Each file stores an **N × 3** array of coordinates that track how a
person drew a digit in 3D space.

-   Length N varies per sample (typically **30--200** time steps)
-   Columns: **x(t), y(t), z(t)**


------------------------------------------------------------------------

## **Step 2 --- Explore the Data**

Run:

``` bash
python explore_data.py
```

This script: 
- loads a few sample motions 
- prints motion shapes and preview rows 
- plots: - 3D motion x(t), y(t), z(t) curves

------------------------------------------------------------------------

## **Step 3 --- Feature Extraction**

File: `extract_features.py`

The raw (N × 3) motion signal is transformed into a fixed-length feature
vector through:

1.  Normalizing the starting point
2.  Normalizing scale
3.  Resampling to a fixed length (e.g., 30 points)
4.  Flattening (30 × 3) → a 90-dimensional vector

This makes sequences comparable for kNN.

------------------------------------------------------------------------

## **Step 4 --- Baseline kNN Classifier**

Run:

    python train_knn.py

This script:

-   loads all CSV files
-   extracts features
-   splits train/validation
-   trains your custom kNN
-   prints accuracy + confusion matrix

Baseline accuracy ≈ **95%**

------------------------------------------------------------------------

## **Step 5 --- Hyperparameter Tuning**

Run:

    python tune_knn.py

Searches:

-   `num_points` ∈ {30, 50, 70}\
-   `k` ∈ {1, 3, 5, 7, 9}

Best: **96%**, with `num_points = 30`, `k = 7`.

------------------------------------------------------------------------

## **Step 6 --- Train Final Model**

Run:

    python train_final_knn.py

Produces file:

    code/knn_model.npz

Contains:

-   X_train (1000 × 90)
-   y_train
-   num_points = 30
-   k = 7

------------------------------------------------------------------------

## **Step 7 --- Final Digit Classifier**

The required function:

``` python
digit_classify(testdata) -> int
```

Steps:

1.  Load knn_model.npz
2.  Extract features
3.  Run kNN
4.  Return predicted digit

Test:

    python test_digit_classify.py

------------------------------------------------------------------------

## Utility Script: project_steps.py

This script runs the *entire project pipeline automatically* in the correct order:

1. Explore data
2. Test feature extraction
3. Baseline kNN training
4. Hyperparameter tuning
5. Final kNN training (saving the model)
6. Testing the final digit classifier

I added this script to make it easy to run all project steps from one place.  
It helps verify that every component of the system works correctly together.

Each individual file (such as `explore_data.py`, `train_knn.py`, or `tune_knn.py`) contains an optional internal `DEBUG` flag.
If you set these flags to **True**, you can view additional results such as printed information, plots, and detailed intermediate outputs in every step of the pipeline.

This script is **only for development and debugging**.  
It is **NOT** used for grading or evaluation.

------------------------------------------------------------------------

# Requirements

    pip install numpy pandas matplotlib

------------------------------------------------------------------------

### DEBUG Flag

Inside scripts such as `explore_data.py` you will find a flag like:

```python
DEBUG = False
```

Setting this flag to True enables extended diagnostic output during execution.
When enabled, the script will:
-	print additional information about the loaded data
-	display intermediate results
-	show extra visualizations and plots
-	provide more detailed step-by-step processing logs

This is especially useful during development and debugging.
For normal usage or when these files are imported by other scripts,
DEBUG should remain False to keep the output clean.

------------------------------------------------------------------------

# Project Complete

This project now fully supports:

-   reading and exploring 3D motion data
-   visualizing sample motions (if needed)
-   extracting normalized features
-   training & tuning custom kNN
-   saving final model
-   predicting digits via `digit_classify()`

------------------------------------------------------------------------
