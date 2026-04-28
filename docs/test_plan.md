# Test Plan

## Scope

The test plan validates feature engineering, training outputs, and API behavior.

## Test cases

1. Feature engineering creates expected derived columns.
2. Churn heuristic returns binary labels.
3. API health endpoint returns success.
4. API readiness endpoint reflects model artifact presence.
5. API prediction returns CLV, churn probability, and drift list.

## Acceptance criteria

- All unit tests pass.
- Model training produces the expected artifact files.
- The API accepts a valid payload and returns a prediction without server errors.
- Monitoring endpoint exposes Prometheus-formatted metrics.

## Test report format

- Total test cases
- Passed
- Failed
- Blocking issues

