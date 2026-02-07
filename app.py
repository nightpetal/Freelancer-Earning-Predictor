from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

saved = joblib.load("model.pkl")
models = saved["models"]
columns = saved["columns"]


@app.route("/", methods=["GET", "POST"])
def index():
    predictions = None

    if request.method == "POST":
        input_data = {
            "Experience_Level_encoded": int(request.form["experience"]),
            "Project_Typeencoded": int(request.form["project_type"]),
            "Job_Category": request.form["job_category"],
            "Platform": request.form["platform"],
            "Client_Region": request.form["client_region"],
            "Hours_Worked_Per_Week": float(request.form["hours"]),
        }

        df = pd.DataFrame([input_data])

        df = pd.get_dummies(df)

        df = df.reindex(columns=columns, fill_value=0)

        predictions = {
            target: round(models[target].predict(df)[0], 2) for target in models
        }

    return render_template("index.html", predictions=predictions)


if __name__ == "__main__":
    app.run(debug=True)
