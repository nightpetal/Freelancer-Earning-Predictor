import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

drop_cols = [
    "Freelancer_ID",
    "Payment_Method",
    "Marketing_Spend",
    "Rehire_Rate",
    "Earnings_USD",
]

y_cols = ["Hourly_Rate", "Job_Success_Rate", "Client_Rating"]

data = pd.read_csv("freelancer_earnings_bd.csv")
data.drop(columns=drop_cols, inplace=True)

data["Experience_Level"] = pd.Categorical(
    data["Experience_Level"],
    categories=["Beginner", "Intermediate", "Expert"],
    ordered=True,
)

le = LabelEncoder()
data["Experience_Level_encoded"] = le.fit_transform(data["Experience_Level"])
data = data.drop(columns=["Experience_Level"])

X = data.drop(columns=y_cols)
y = data[y_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=20
)

nominal_cols = ["Job_Category", "Platform", "Client_Region", "Project_Type"]

# One-hot encode training set
X_train = pd.get_dummies(X_train, columns=nominal_cols, drop_first=True)

X_test = pd.get_dummies(X_test, columns=nominal_cols, drop_first=True)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Predictions:\n", predictions)

for i, col in enumerate(y_cols):
    r2 = r2_score(y_test[col], predictions[:, i])
    print(f"R² score for {col}: {r2:.4f}")
