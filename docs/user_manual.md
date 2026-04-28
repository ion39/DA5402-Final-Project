# User Manual

## What this application does

This application estimates the likely lifetime value of an individual customer and highlights whether the customer appears to be at elevated churn risk.

## How to use it

1. Start the backend API.
2. Start the frontend server or open the frontend through Docker Compose.
3. Open the console in your browser.
4. Click `Load sample` to populate a valid customer profile or enter values manually.
5. Click `Estimate CLV`.
6. Review the predicted CLV, churn probability, and drift summary.

## Operational panels

- Prediction section:
  enter or load a customer profile and run inference.
- Pipeline visibility section:
  review training status, row counts, and model quality summary.

## For non-technical users

- Higher predicted CLV means the customer is expected to generate more value.
- Higher churn probability means the customer may need attention from retention programs.
- Drifted features mean the current input looks meaningfully different from the data the model was trained on.

