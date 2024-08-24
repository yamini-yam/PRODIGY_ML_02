**Introduction**

Welcome to my customer clustering project! The goal of this project is to segment customers into different types based on their purchase behavior using a clustering algorithm. The dataset includes various features related to customer transactions and spending patterns. In this project, I've used K-Means clustering to categorize customers into distinct groups and developed an interactive web application with Streamlit to visualize these segments.

**Data Overview**

The dataset used in this project includes the following features:

**Total Purchase Amount:** The total amount spent by the customer.

**Number of Transactions:** The number of transactions made by the customer.

**Frequency of Purchases:** How often the customer makes purchases.

**Recency of Last Purchase:** The number of days since the customer's last purchase.

**Electronics Purchases:** The percentage of the total purchase amount spent on electronics.

**Clothing Purchases:** The percentage of the total purchase amount spent on clothing.

**Grocery Purchases:** The percentage of the total purchase amount spent on groceries.

**Discount Sensitivity:** The customer's sensitivity to discounts (converted from percentage).

I performed basic data preprocessing, including converting percentage strings to floats, encoding categorical variables, and standardizing the feature values.

**Model Development**

For this project, I implemented the following steps:

**Data Preparation:** Loaded and preprocessed the dataset.

**Feature Selection:** Chose relevant features for clustering.

**Standardization:** Standardized the features using StandardScaler.

**Model Training:** Applied K-Means clustering to segment customers into clusters.

**Cluster Mapping:** Mapped clusters to meaningful customer types for easier interpretation.

The K-Means algorithm was used to identify three distinct customer segments, and these segments were labeled as 'Regular', 'Occasional', and 'Rare'.

**Web Application**

An interactive web application was developed using Streamlit. The application allows users to input various customer features and obtain predictions on the customer type. The Streamlit interface is designed to be intuitive and provides real-time clustering results based on the input data.

**Future Enhancements**

This initial project can be enhanced in several ways:

**Feature Engineering:** Explore additional features or transformations to improve clustering results.

**Model Enhancement:** Experiment with different clustering algorithms or parameters to find the optimal solution.

**User Interface:** Add more features or visualizations to the Streamlit app to enhance the user experience.

**Conclusion**

This project demonstrates how clustering techniques can be applied to customer segmentation based on purchase behavior. By leveraging K-Means clustering and developing an interactive web application, I've created a practical tool for understanding customer types and behaviors.

**Acknowledgement**

This project was completed as part of the @Prodigy_Infotech internship program.

Feel free to explore the project, test the web application, and modify the code as needed. If you have any questions or feedback, please reach out to me at yaminirameshkumar94@gmail.com.
