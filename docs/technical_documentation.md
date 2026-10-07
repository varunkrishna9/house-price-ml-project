# Technical Documentation

## Architecture
CSV → validation → feature engineering → preprocessing → model comparison → cross-validation → explainability → model artifact → Flask API/UI.

## Evaluation
MAE measures average absolute error, RMSE penalizes larger errors, R² measures explained variance, and MAPE measures percentage error.

## Error Handling
Input validation checks required fields, numeric ranges and allowed categories. Client errors return HTTP 400; unexpected failures return HTTP 500.

## Testing
Tests cover feature engineering, successful prediction and invalid input handling.

## Security
Do not expose the development server directly to the public internet. Add HTTPS, authentication, rate limiting and production WSGI serving before deployment.
