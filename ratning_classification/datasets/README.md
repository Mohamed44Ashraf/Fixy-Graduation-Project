# Home Service Technician Performance & Customer Satisfaction Dataset

## Overview
This dataset contains **300,000 service records** focused on analyzing **technician performance**, **customer satisfaction**, and overall **service quality**.

It includes multiple rating dimensions such as punctuality, professionalism, communication, handling quality, and cost satisfaction.

The dataset is designed to support:
- Machine Learning projects
- Exploratory Data Analysis (EDA)
- Business insights and dashboard creation
- Rating prediction systems

## Dataset Information
- Rows: 300,000
- Columns: 20
- Format: CSV

## Column Dictionary
| Column Name | Description |
|------------|------------|
| technician_id | Unique identifier for the technician |
| customer_id | Unique identifier for the customer |
| service_type | Type of service provided |
| day_of_week | Day when the service was performed |
| time_of_day | Time period of the service |
| weather_condition | Weather condition during the service |
| job_duration_minutes | Total service duration in minutes |
| response_time_minutes | Response time before service started |
| punctuality_rating | Rating for technician punctuality |
| professionalism_rating | Rating for professionalism |
| communication_rating | Rating for communication quality |
| handling_quality | Rating for execution quality |
| cost_satisfaction | Satisfaction regarding cost |
| customer_satisfaction | Overall customer satisfaction |
| customer_behavior | Customer behavior category/level |
| clarity_of_issue | Issue clarity score |
| safety_and_environment | Safety/environment score |
| customer_punctuality | Customer punctuality score |
| worker_satisfaction | Technician satisfaction score |
| final_rating | Final overall rating (Target Variable) |

## Target Variable
The main prediction target is:
- `final_rating`

## Suggested Machine Learning Tasks
- **Regression**: Predict `final_rating`
- **Classification**: Convert `final_rating` into categories (low/medium/high)
- Customer satisfaction prediction
- Technician performance scoring system

## Possible Insights
- Factors affecting customer satisfaction
- Impact of response time and job duration on final rating
- Comparing service types and performance
- Relationship between worker satisfaction and customer satisfaction

## Example Usage
```python
import pandas as pd

df = pd.read_csv("service_feedback_dataset.csv")
print(df.head())
print(df.describe())