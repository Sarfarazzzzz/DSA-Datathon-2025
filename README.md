# Food Desert Prediction & Prevention (DSA Datathon 2025)

## 🏆 Datathon 2nd Runner-Up (3nd Place) 🏆

This project was developed for the **DSA Datathon 2025**, where it achieved **Runner-Up (2nd Place)** out of all participating teams.

## 1. Project Goal

The primary goal of this 24-hour datathon was to build a machine learning model to identify U.S. census tracts at high risk of becoming "food deserts".

Beyond just prediction, the project aims to identify the key socioeconomic and health factors that contribute to food insecurity and, based on these findings, propose actionable, data-driven policy recommendations to prevent the formation of new food deserts.

## 2. Tech Stack

* **Data Analysis & Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn, XGBoost, CatBoost

## 3. Project Workflow

1.  **Data Integration:** Combined and cleaned over 10 disparate datasets from sources like the Food Access Research Atlas and the Food Environment Atlas, merging them at the census-tract level.
2.  **Exploratory Data Analysis (EDA):** Performed extensive EDA to handle missing values and uncover correlations between food access, socioeconomic status (e.g., poverty, unemployment), and health outcomes (e.g., obesity, diabetes).
3.  **Feature Engineering:** Engineered new features from the raw data to better capture the complex factors leading to food insecurity.
4.  **Model Development:** Trained and compared a suite of classification models, including Logistic Regression, Random Forest, XGBoost, and CatBoost, to find the best predictor.
5.  **Model Evaluation:** The final tuned model achieved **88% accuracy** in predicting at-risk census tracts.

## 4. Key Findings & Policy Recommendations

Based on the model's feature importances and our analysis, we proposed a multi-pronged policy solution to address the root causes of food deserts:

* **Economic Incentives:** Provide tax incentives and grants to encourage new grocery stores and farmers' markets to open in identified at-risk areas.
* **Logistical Support:** Fund and expand mobile market programs and invest in public transportation to connect residents of at-risk tracts to existing fresh food retailers.
* **Community Engagement:** Partner with local organizations to establish and support community-supported agriculture (CSA) programs and urban gardens.

## 5. How to Run This Project

1.  Clone this repository:
    ```bash
    git clone [https://github.com/Sarfarazzzzz/DSA-Datathon-2025.git](https://github.com/Sarfarazzzzz/DSA-Datathon-2025.git)
    ```
2.  Install the required dependencies:
    *(**Note:** You should create a `requirements.txt` file listing all the libraries used in `Final.py`)*
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the Python script to perform the analysis and model training:
    ```bash
    python Final.py
    ```
