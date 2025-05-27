# 🚀 Credit Risk Management Model for Banking

## 📊 Project Overview

This project focuses on developing and training a robust **Credit Risk Management Model** designed to assess the creditworthiness of loan applicants for a banking institution. Leveraging a comprehensive dataset, the model aims to minimize potential losses due to loan defaults by accurately predicting the probability of default for individual customers.

**Key Goals:**
* **Accurate Default Prediction:** Build a predictive model to identify high-risk applicants.
* **Feature Importance Analysis:** Understand which factors significantly influence credit risk.
* **Model Interpretability:** Provide insights into why a specific decision was made.
* **Risk Mitigation:** Empower banks to make informed lending decisions, thereby reducing Non-Performing Loans (NPLs).

## ✨ Features

* **Data Preprocessing:** Handling of missing values, encoding categorical features (e.g., using One-Hot Encoding), and scaling numerical data.
* **Feature Engineering:** Creation of new, more informative features from raw data to improve model performance, including techniques like binning, aggregation, and interaction terms.
* **Robust Model Training & Selection:**
    * Primary model: **CatBoost** — chosen for its superior performance, handling of categorical features, and robustness.
    * Comparative models: **LightGBM** and **XGBoost** were also utilized for benchmarking and performance comparison.
* **Hyperparameter Tuning:** Optimization of model parameters for best performance.
* **Model Evaluation:** Comprehensive evaluation on a held-out test set using metrics like AUC-ROC.
* **Probability Distribution Analysis:** Visualization of predicted probabilities for better risk segmentation.

## 🛠️ Technologies Used

* `Python`
* `Pandas`
* `NumPy`
* `Scikit-learn`
* `CatBoost`
* `LightGBM`
* `XGBoost`
* `Matplotlib`
* `IPython`
* `tqdm`
* `pickle`

## 🚀 Installation and Setup

To set up and run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/romank-data/credit-risk-management-model.git
    cd credit-risk-management-model
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the dataset:**
    This project utilizes the [Default Risk dataset from Kaggle](https://www.kaggle.com/datasets/romanwr/train-data). Please download the necessary files (e.g., '.pq' files, 'train_target.csv' and 'credit-pipeline.py') and place them in the appropriate directory as referenced in the `ml-credit-score.ipynb` notebook (e.g., in an `input` folder at the root of your project).

5.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook ml-credit-score.ipynb
    ```
    Open `ml-credit-score.ipynb` in your browser and run all cells to reproduce the analysis and model training.

## 📈 Results and Insights

The **CatBoost model** demonstrates strong predictive capabilities in identifying potential loan defaulters. The probability distribution plot (as shown in the screenshot) visually represents the model's confidence in classifying applicants into different risk categories, allowing for better segmentation and targeted interventions by the bank. Key features influencing credit risk were identified, providing valuable insights for strategic decision-making.

## 🤝 Contribution

Feel free to fork this repository, open issues, or submit pull requests. Any contributions are welcome!

## 📧 Contact

If you have any questions or suggestions, feel free to reach out:

* **Roman Kostenko** - [GitHub Profile](https://github.com/[romank-data])
* **Email:** [roman.kostenko@hotmail.com](mailto:roman.kostenko@hotmail.com)
