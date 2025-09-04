# Campus Financial Aid Modeling

## Project Overview
This project develops a data-driven framework to improve the accuracy and fairness of financial aid allocation in universities. Using campus card transaction data, we aim to identify students with the greatest financial need and design equitable aid distribution strategies.

Key tasks include:

- Exploratory Data Analysis (EDA) of student consumption records  
- Detection of anomalous or abnormally low spending patterns  
- Comprehensive evaluation modeling (TOPSIS + Entropy Weight)  
- Identification of the top 50 students most in need of aid  
- Clustering methods to assign aid levels  
- Estimation of reasonable aid ranges  
- Evaluation of model effectiveness and potential generalization  

---

## Data Description
- **Source:** Campus card transaction dataset  
- **File:** `dataset.xlsx`  
- **Observations:** 10,000+ transaction records across multiple students  
- **Variables:**  
  - **Student ID (校园卡号):** Anonymized student identifier  
  - **Gender (性别):** Male/Female  
  - **Transaction Time (消费时间):** Date and time of each record  
  - **Amount Spent (消费金额):** Value of individual transactions (CNY)  
  - **Deposit Amount (存储金额):** Recharge to card balance  
  - **Balance (余额):** Remaining balance after transaction  
  - **Transaction Count (消费次数):** Running count of transactions  
  - **Transaction Type (消费类型):** e.g., purchase, deposit  
  - **Location (消费地点):** Venue of transaction (e.g., dining halls, shops)  

---

## Analysis Overview
- **Exploratory Analysis:** Summary statistics of transaction frequencies, spending patterns, and distributions across venues.  
- **Anomaly Detection:** Identification of unusually low or irregular transactions using Z-scores, percentiles, or boxplot methods.  
- **Evaluation Model:** Integration of TOPSIS and Entropy Weight to assess financial need levels.  
- **Selection:** Top 50 students identified as candidates for aid.  
- **Clustering:** K-means clustering used to categorize aid levels and estimate ranges.  
- **Model Validation:** Reverse testing and rules for inclusion/exclusion to improve classification accuracy.  

---

## Key Findings
- Students with frequent low-value transactions are more likely to experience financial constraints.  
- Spending patterns by time of day (e.g., breakfast, lunch, dinner) provide signals of economic hardship.  
- The evaluation model effectively prioritizes students in greatest need compared to traditional subjective approaches.  
- Clustering results generate tiered aid recommendations that balance fairness and practicality.  
- The framework demonstrates the potential of big data analytics in advancing equitable education policies.
