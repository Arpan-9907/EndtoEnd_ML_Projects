import os
import sys
import json
import pandas as pd
from flask import Flask, render_template, request, send_file, redirect, url_for
from werkzeug.utils import secure_filename

from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.train_pipeline import TrainingPipeline
from src.logger import logging

app = Flask(__name__)
app.secret_key = "home_credit_secret_key"

UPLOAD_FOLDER = os.path.join("artifacts", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

predict_pipeline = PredictPipeline()

FEATURE_METADATA = {
    "EXT_SOURCE_1": {
        "label": "External Source 1 (Score)",
        "description": "Normalized score from external data source 1 (0.0 - 1.0)",
        "step": "0.001",
        "default": "0.50"
    },
    "EXT_SOURCE_2": {
        "label": "External Source 2 (Score)",
        "description": "Normalized score from external data source 2 (0.0 - 1.0)",
        "step": "0.001",
        "default": "0.50"
    },
    "EXT_SOURCE_3": {
        "label": "External Source 3 (Score)",
        "description": "Normalized score from external data source 3 (0.0 - 1.0)",
        "step": "0.001",
        "default": "0.50"
    },
    "DAYS_BIRTH": {
        "label": "Age in Days (Negative)",
        "description": "Client age in days at application (e.g. -14600 ≈ 40 years old)",
        "step": "1",
        "default": "-14600"
    },
    "DAYS_REGISTRATION": {
        "label": "Days Since Registration (Negative)",
        "description": "Days before application client changed registration (e.g. -4000)",
        "step": "1",
        "default": "-4500"
    },
    "DAYS_ID_PUBLISH": {
        "label": "Days Since ID Publish (Negative)",
        "description": "Days before application client changed ID document (e.g. -2500)",
        "step": "1",
        "default": "-2500"
    },
    "AMT_ANNUITY": {
        "label": "Loan Annuity Amount",
        "description": "Monthly loan installment/annuity amount (e.g. 25000)",
        "step": "100",
        "default": "25000"
    },
    "AMT_GOODS_PRICE": {
        "label": "Goods Price Amount",
        "description": "Price of the goods for which the credit is given (e.g. 450000)",
        "step": "1000",
        "default": "450000"
    },
    "FLOORSMIN_AVG": {
        "label": "Building Minimum Floors (Avg)",
        "description": "Normalized minimum floors of client building (0.0 - 1.0)",
        "step": "0.01",
        "default": "0.15"
    },
    "AMT_REQ_CREDIT_BUREAU_HOUR": {
        "label": "Credit Bureau Inquiries (1 Hour)",
        "description": "Number of inquiries to Credit Bureau 1 hour before application",
        "step": "1",
        "default": "0"
    }
}


def get_top_features_and_means():
    top_features = []
    feature_means = {}
    top_features_path = os.path.join("artifacts", "top_features.json")
    if os.path.exists(top_features_path):
        try:
            with open(top_features_path, "r") as f:
                data = json.load(f)
                top_features = data.get("top_features", [])
                feature_means = data.get("feature_means", {})
        except Exception as e:
            logging.error(f"Error loading top_features.json: {e}")

    if not top_features:
        top_features = list(FEATURE_METADATA.keys())

    return top_features, feature_means


@app.route("/", methods=["GET", "POST"])
@app.route("/predict", methods=["GET", "POST"])
def predict():
    top_features, feature_means = get_top_features_and_means()
    prediction = None
    prediction_label = None
    prediction_class = None
    user_inputs = {}

    if request.method == "POST":
        try:
            user_data = {}
            for key, val in request.form.items():
                clean_val = val.strip()
                if clean_val != "":
                    try:
                        user_data[key] = float(clean_val)
                    except ValueError:
                        user_data[key] = clean_val
            user_inputs = user_data

            pred = predict_pipeline.predict_from_dict(user_data)
            # Model mapping: {0: 'bad', 1: 'good'}
            target_mapping = {0: "bad", 1: "good"}
            mapped_pred = target_mapping.get(pred, str(pred))
            prediction = mapped_pred

            if mapped_pred == "good" or pred == 1:
                prediction_label = "Good Credit - Low Risk (Eligible for Approval)"
                prediction_class = "success"
            else:
                prediction_label = "Bad Credit - High Default Risk (Caution / High Risk)"
                prediction_class = "danger"

            logging.info(f"Single prediction successful. Result: {prediction}")
            return render_template(
                "form.html",
                top_features=top_features,
                feature_means=feature_means,
                feature_metadata=FEATURE_METADATA,
                prediction=prediction,
                prediction_label=prediction_label,
                prediction_class=prediction_class,
                user_inputs=user_inputs
            )

        except Exception as e:
            logging.error(f"Error in single prediction: {e}")
            return render_template(
                "form.html",
                top_features=top_features,
                feature_means=feature_means,
                feature_metadata=FEATURE_METADATA,
                error=f"Prediction failed: {str(e)}",
                user_inputs=user_inputs
            )

    return render_template(
        "form.html",
        top_features=top_features,
        feature_means=feature_means,
        feature_metadata=FEATURE_METADATA,
        user_inputs=user_inputs
    )


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        try:
            if "file" not in request.files:
                return render_template("upload.html", error="No file part in the request.")

            file = request.files["file"]
            if file.filename == "":
                return render_template("upload.html", error="No file selected. Please choose a CSV file.")

            if not file.filename.lower().endswith(".csv"):
                return render_template("upload.html", error="Invalid file format. Please upload a .csv file.")

            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)
            logging.info(f"File uploaded successfully to {file_path}")

            output_path = predict_pipeline.predict_from_csv(file_path)
            logging.info(f"Batch prediction completed: {output_path}")

            pred_df = pd.read_csv(output_path)
            total_records = len(pred_df)
            counts = pred_df["TARGET"].value_counts().to_dict() if "TARGET" in pred_df.columns else {}

            # First 10 rows preview
            preview_cols = [c for c in ["TARGET"] + [col for col in pred_df.columns if col != "TARGET"]][:12]
            preview_df = pred_df[preview_cols].head(10)
            preview_records = preview_df.to_dict(orient="records")

            return render_template(
                "upload.html",
                success=True,
                total_records=total_records,
                counts=counts,
                preview_records=preview_records,
                preview_cols=preview_cols,
                filename=filename
            )

        except Exception as e:
            logging.error(f"Error during batch prediction: {e}")
            return render_template("upload.html", error=f"Batch prediction failed: {str(e)}")

    return render_template("upload.html")


@app.route("/download")
def download():
    prediction_file_path = os.path.join("artifacts", "predictions", "prediction_file.csv")
    if os.path.exists(prediction_file_path):
        return send_file(
            prediction_file_path,
            as_attachment=True,
            download_name="prediction_file.csv",
            mimetype="text/csv"
        )
    return "Prediction file not found. Please run batch prediction first.", 404


@app.route("/train", methods=["GET", "POST"])
def train():
    if request.method == "POST":
        try:
            training_pipeline = TrainingPipeline()
            model_path = training_pipeline.run_pipeline()
            return render_template(
                "upload.html",
                train_success=True,
                train_message=f"Model retrained successfully and saved at: {model_path}"
            )
        except Exception as e:
            return render_template("upload.html", train_error=f"Training failed: {str(e)}")
    return redirect(url_for("predict"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)