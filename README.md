# Strava Prediction

This project uses the Strava API to fetch activity data for a user and predict whether the user will be active in the next 8 hours. Additionally, it predicts the number of likes or kudos based on the number of reeards.

## Project Description

The project includes automated data fetching, preprocessing, model training, and prediction using GitHub Actions. Results are accessible via an API and a UI application.

## Functionalities

- **Data Fetching**: Automated data retrieval from the Strava API.
- **Preprocessing and Modeling**: Data preprocessing and model training for predictions.
- **API**: Access to predictions via an API.
- **UI Application**: Display of predictions in a UI application.

## GitHub Actions

The project leverages GitHub Actions to automate the following tasks:
- Fetching data from the Strava API.
- Preprocessing the data.
- Training and testing the model.
- Saving the model if it performs better.
- Generating daily predictions.

## Model Management with DagsHub

Models and their associated metrics are stored on DagsHub. Better-performing models are selected based on these metrics, ensuring that the most accurate models are used for predictions.

## Example

  ![admin](https://github.com/user-attachments/assets/c949fb68-e4e0-434e-8b4b-72ce1ec56d36)
  ![image](https://github.com/user-attachments/assets/61aa777c-3d80-4f96-8a9a-9c44f434511b)
  
  


  
