from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = Path(__file__).resolve().parent / "optimized_salary_model.pkl"

# Load the serialized RandomForestRegressor trained on the engineered feature vector.
model = joblib.load(MODEL_PATH)

FEATURE_COLUMNS = [
    "Age",
    "Performance_Score",
    "Remote_Work",
    "Join_Year",
    "Join_Month",
    "Experience_Years",
    "Department_Cloud Tech",
    "Department_DevOps",
    "Department_Finance",
    "Department_HR",
    "Department_Sales",
    "Region_Florida",
    "Region_Illinois",
    "Region_Nevada",
    "Region_New York",
    "Region_Texas",
    "Status_Inactive",
    "Status_Pending",
]

DEPARTMENT_OPTIONS = ["Active", "Cloud Tech", "DevOps", "Finance", "HR", "Sales"]
REGION_OPTIONS = ["California", "Florida", "Illinois", "Nevada", "New York", "Texas"]
STATUS_OPTIONS = ["Active", "Inactive", "Pending"]


def _parse_int(value, field_name, minimum, maximum):
    try:
        parsed_value = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be a whole number.") from None

    if not minimum <= parsed_value <= maximum:
        raise ValueError(f"{field_name} must be between {minimum} and {maximum}.")
    return parsed_value


def _parse_float(value, field_name, minimum, maximum):
    try:
        parsed_value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be a number.") from None

    if not minimum <= parsed_value <= maximum:
        raise ValueError(f"{field_name} must be between {minimum} and {maximum}.")
    return parsed_value


def build_feature_frame(form_data):
    """Create a single-row DataFrame that matches the training feature order exactly."""
    age = _parse_int(form_data.get("age", 0), "Age", 18, 80)
    performance_score = _parse_float(form_data.get("performance_score", 0), "Performance Score", 1, 4)
    remote_work = _parse_int(form_data.get("remote_work", 0), "Remote Work", 0, 1)
    join_year = _parse_int(form_data.get("join_year", 0), "Join Year", 2000, 2030)
    join_month = _parse_int(form_data.get("join_month", 0), "Join Month", 1, 12)
    experience_years = _parse_float(form_data.get("experience_years", 0), "Experience Years", 0, 40)

    # Dummy-variable mapping must mirror the original get_dummies(drop_first=True) logic.
    # The baseline categories are: Department = 'Active', Region = 'California', Status = 'Active'.
    # When the user selects one of those baseline values, every dummy column for that family stays 0.
    department = form_data.get("department", "Active")
    region = form_data.get("region", "California")
    status = form_data.get("status", "Active")

    department_dummies = {
        "Department_Cloud Tech": 0,
        "Department_DevOps": 0,
        "Department_Finance": 0,
        "Department_HR": 0,
        "Department_Sales": 0,
    }
    if department in department_dummies:
        department_dummies[f"Department_{department}"] = 1

    region_dummies = {
        "Region_Florida": 0,
        "Region_Illinois": 0,
        "Region_Nevada": 0,
        "Region_New York": 0,
        "Region_Texas": 0,
    }
    if region in {"Florida", "Illinois", "Nevada", "New York", "Texas"}:
        region_dummies[f"Region_{region}"] = 1

    status_dummies = {
        "Status_Inactive": 0,
        "Status_Pending": 0,
    }
    if status == "Inactive":
        status_dummies["Status_Inactive"] = 1
    elif status == "Pending":
        status_dummies["Status_Pending"] = 1

    feature_values = [
        age,
        performance_score,
        remote_work,
        join_year,
        join_month,
        experience_years,
    ]
    feature_values.extend(
        [department_dummies["Department_Cloud Tech"], department_dummies["Department_DevOps"], department_dummies["Department_Finance"], department_dummies["Department_HR"], department_dummies["Department_Sales"]]
    )
    feature_values.extend(
        [region_dummies["Region_Florida"], region_dummies["Region_Illinois"], region_dummies["Region_Nevada"], region_dummies["Region_New York"], region_dummies["Region_Texas"]]
    )
    feature_values.extend([status_dummies["Status_Inactive"], status_dummies["Status_Pending"]])

    return pd.DataFrame([feature_values], columns=FEATURE_COLUMNS)


def predict_salary(form_data):
    feature_frame = build_feature_frame(form_data)
    prediction = model.predict(feature_frame)[0]
    return round(float(prediction), 2)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    prediction_display = None
    error_message = None

    if request.method == "POST":
        try:
            prediction = predict_salary(request.form)
            prediction_display = f"${prediction:,.2f}"
        except ValueError as exc:
            error_message = str(exc)
        except Exception as exc:  # pragma: no cover - defensive runtime protection
            error_message = f"Prediction failed: {exc}"

    return render_template(
        "index.html",
        prediction=prediction,
        prediction_display=prediction_display,
        error_message=error_message,
        form=request.form,
        department_options=DEPARTMENT_OPTIONS,
        region_options=REGION_OPTIONS,
        status_options=STATUS_OPTIONS,
    )


if __name__ == "__main__":
    app.run(debug=True)
