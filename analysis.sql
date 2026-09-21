-- Example analytical queries for the customer churn project

-- Churn rate
SELECT
    ROUND(100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2)
        AS churn_rate_percent
FROM customer_churn;

-- Churn by contract
SELECT
    contract,
    COUNT(*) AS customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers
FROM customer_churn
GROUP BY contract
ORDER BY churned_customers DESC;

-- Average monthly charges by churn status
SELECT
    churn,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
FROM customer_churn
GROUP BY churn;
