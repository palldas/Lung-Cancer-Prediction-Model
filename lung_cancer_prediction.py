import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("survey lung cancer.csv")

# Separate features and target variable
X = data.drop(columns='LUNG_CANCER')
y = data['LUNG_CANCER']

# Define categorical and numerical features
categorical_features = ['GENDER']
binary_categorical_features = [
    'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE', 
    'FATIGUE ', 'ALLERGY ', 'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 
    'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN'
]
numerical_features = ['AGE']

# Define the ColumnTransformer for data preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),  # OneHotEncode categorical features
        ('num', StandardScaler(), numerical_features),   # Standardize numerical features
        ('bin', 'passthrough', binary_categorical_features)  # Keep binary features as is
    ]
)

# Apply the transformations
X_transformed = preprocessor.fit_transform(X)

# Get column names after transformation
cat_columns = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_columns = list(cat_columns) + numerical_features + binary_categorical_features

# Create a DataFrame with the transformed data
X_transformed_df = pd.DataFrame(X_transformed, columns=all_columns)

# Encode the target variable (LUNG_CANCER)
y_encoded = y.apply(lambda x: 1 if x == 'YES' else 0)

# Split data into training and validation sets manually
train_fraction = 0.5
train = X_transformed_df.sample(frac=train_fraction, random_state=42)
val = X_transformed_df.drop(train.index)
y_train = y_encoded.loc[train.index]
y_val = y_encoded.loc[val.index]

# -------------------------
# Linear Regression
# -------------------------

# Initialize and train Linear Regression model
model_lr = LinearRegression()
model_lr.fit(train, y_train)

# Predict and evaluate Linear Regression model
y_pred_lr = model_lr.predict(val)
rmse_lr = mean_squared_error(y_val, y_pred_lr, squared=False)
mae_lr = mean_absolute_error(y_val, y_pred_lr)

print(f"Linear Regression RMSE: {rmse_lr}")
print(f"Linear Regression MAE: {mae_lr}")

# Loadings plot for Linear Regression
coefficients = model_lr.coef_
coef_df = pd.DataFrame({'Feature': all_columns, 'Coefficient': coefficients})
coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values(by='Abs_Coefficient', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(coef_df['Feature'], coef_df['Coefficient'])
plt.xlabel('Coefficient')
plt.title('Loadings Plot')
plt.show()

# -------------------------
# K-Nearest Neighbors Regression
# -------------------------

# Hyperparameter tuning using GridSearchCV for KNN Regression
param_grid = {'n_neighbors': range(1, 21)}
grid_search_knn = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search_knn.fit(train, y_train)

# Retrieve the optimal k value
best_k = grid_search_knn.best_params_['n_neighbors']

# Train KNN Regression model with the optimal k
model_knn_reg = KNeighborsRegressor(n_neighbors=best_k)
model_knn_reg.fit(train, y_train)

# Predict and evaluate KNN Regression model
y_pred_knn_reg = model_knn_reg.predict(val)
rmse_knn_reg = mean_squared_error(y_val, y_pred_knn_reg, squared=False)
mae_knn_reg = mean_absolute_error(y_val, y_pred_knn_reg)

print(f"KNN Regression RMSE: {rmse_knn_reg}")
print(f"KNN Regression MAE: {mae_knn_reg}")

# -------------------------
# K-Nearest Neighbors Classification
# -------------------------

# Hyperparameter tuning using GridSearchCV for KNN Classification
param_grid = {'n_neighbors': range(1, 21)}
grid_search_knn_cls = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search_knn_cls.fit(train, y_train)

# Retrieve the optimal k value
best_k_cls = grid_search_knn_cls.best_params_['n_neighbors']

# Train KNN Classification model with the optimal k
model_knn_cls = KNeighborsClassifier(n_neighbors=best_k_cls)
model_knn_cls.fit(train, y_train)

# Predict and evaluate KNN Classification model
y_pred_knn_cls = model_knn_cls.predict(val)
accuracy_knn_cls = accuracy_score(y_val, y_pred_knn_cls)
precision_knn_cls = precision_score(y_val, y_pred_knn_cls)
recall_knn_cls = recall_score(y_val, y_pred_knn_cls)
f1_knn_cls = f1_score(y_val, y_pred_knn_cls)

# Calculate specificity
tn, fp, fn, tp = confusion_matrix(y_val, y_pred_knn_cls).ravel()
specificity_knn_cls = tn / (tn + fp)

print(f"KNN Classification Accuracy: {accuracy_knn_cls}")
print(f"KNN Classification Precision: {precision_knn_cls}")
print(f"KNN Classification Recall: {recall_knn_cls}")
print(f"KNN Classification F1-Score: {f1_knn_cls}")
print(f"KNN Classification Specificity: {specificity_knn_cls}")