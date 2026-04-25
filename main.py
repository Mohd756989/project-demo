import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

df = pd.read_csv("laptop.csv")

df.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1, inplace=True)
print(df.head())
df["Company"] = df["Company"].fillna(df["Company"].mode()[0])
df["Company"].isnull().sum()

df["TypeName"] = df["TypeName"].fillna(df["TypeName"].mode()[0])
df["TypeName"].unique()

df["Inches"] = df["Inches"].replace('?', np.nan)

df["Inches"] = df["Inches"].astype(float)
df["Inches"] = df["Inches"].fillna(df["Inches"].median())

df["ScreenResolution"] = df["ScreenResolution"].replace('?',np.nan)
df["ScreenResolution"] = df["ScreenResolution"].fillna(df["ScreenResolution"].mode()[0])
df["ScreenResolution"].isnull().sum()

df["Cpu"] = df["Cpu"].replace('?',np.nan)
df["Cpu"] = df["Cpu"].fillna(df["Cpu"].mode()[0])
df["Cpu"].isnull().sum()

df["Ram"] = df["Ram"].fillna(df["Ram"].mode()[0])
df["Ram"] = df["Ram"].str.replace('GB','')
df["Ram"] = df["Ram"].astype(int)
df["Ram"].unique()

df["Weight"] = df['Weight'].replace('?',np.nan)
df["Weight"] = df["Weight"].fillna(df["Weight"].mode()[0])
df["Weight"] = df["Weight"].str.replace('kg','').astype(float)
df["Weight"].unique()

df["Gpu"] = df["Gpu"].replace('?',np.nan)
df["Gpu"] = df["Gpu"].fillna(df["Gpu"].mode()[0])
df["Gpu"].isnull().sum()

df["Memory"] = df["Memory"].replace('?',np.nan)
df["Memory"] = df["Memory"].fillna(df["Memory"].mode()[0])
df["Memory"].isnull().sum()
def parse_memory(mem):
    # This is a simple parser; you might need to enhance it
    mem = mem.replace('GB', '').replace('TB', '000')
    # For simplicity, take the first number
    import re
    nums = re.findall(r'\d+', mem)
    if nums:
        return int(nums[0])
    return 0

df['Memory_GB'] = df['Memory'].apply(parse_memory)

df["OpSys"] = df["OpSys"].fillna(df["OpSys"].mode()[0])
df["OpSys"].isnull().sum()

df["Price"] = df["Price"].fillna(df["Price"].median())
df["Price"].isnull().sum()

Q1 = df['Weight'].quantile(0.25)
Q3 = df['Weight'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df['Weight'] = df['Weight'].clip(lower, upper)


Q1 = df['Inches'].quantile(0.25)
Q3 = df['Inches'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df['Inches'] = df['Inches'].clip(lower, upper)


Q1 = df['Ram'].quantile(0.25)
Q3 = df['Ram'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df['Ram'] = df['Ram'].clip(lower, upper)

Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df['Price'] = df['Price'].clip(lower, upper)

from sklearn.preprocessing import LabelEncoder
le_company = LabelEncoder()
df['Company_encoded'] = le_company.fit_transform(df['Company'])

le_typename = LabelEncoder()
df['TypeName_encoded'] = le_typename.fit_transform(df['TypeName'])

le_cpu = LabelEncoder()
df['Cpu_encoded'] = le_cpu.fit_transform(df['Cpu'])

le_gpu = LabelEncoder()
df['Gpu_encoded'] = le_gpu.fit_transform(df['Gpu'])

le_opsys = LabelEncoder()
df['OpSys_encoded'] = le_opsys.fit_transform(df['OpSys'])

features = ['Inches', 'Ram', 'Memory_GB', 'Weight', 'Company_encoded', 'TypeName_encoded', 'Cpu_encoded', 'Gpu_encoded', 'OpSys_encoded']
X = df[features]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
model = XGBRegressor(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8
)
# model = LinearRegression()
model.fit(X_train, y_train)
# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
print(f'Mean Absolute Error: {mae}')


# Save the model and encoders
import joblib
joblib.dump(model, 'xg_boost_model.pkl')
joblib.dump(le_company, 'le_company.pkl')
joblib.dump(le_typename, 'le_typename.pkl')
joblib.dump(le_cpu, 'le_cpu.pkl')
joblib.dump(le_gpu, 'le_gpu.pkl')
joblib.dump(le_opsys, 'le_opsys.pkl')
