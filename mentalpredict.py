import numpy as np
import pandas as pd
import sns as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('./Student Mental health.csv')

# Preprocess the data
df['Do you have Depression?'] = df['Do you have Depression?'].map({'Yes': 1, 'No': 0})

# Select relevant features
features = ['Choose your gender', 'Age', 'Your current year of Study', 'Do you have Anxiety?', 'Do you have Panic attack?']

# Split the data into input features (X) and target variable (y)
X = df[features]
y = df['Do you have Depression?']

# Perform one-hot encoding for categorical features
X_encoded = pd.get_dummies(X)

# Handle missing values
imputer = SimpleImputer(strategy='most_frequent')
X_encoded_filled = pd.DataFrame(imputer.fit_transform(X_encoded), columns=X_encoded.columns)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded_filled, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prompt user for input
gender = input("Choose your gender (Male/Female): ")
age = int(input("Enter your age: "))
year_of_study = input("Enter your current year of study: ")
has_anxiety = input("Do you have anxiety? (Yes/No): ")
has_panic_attack = input("Do you have panic attacks? (Yes/No): ")

# Create a dictionary with user input
user_input = {
    'Choose your gender': gender,
    'Age': age,
    'Your current year of Study': year_of_study,
    'Do you have Anxiety?': has_anxiety,
    'Do you have Panic attack?': has_panic_attack
}

# Create a DataFrame from the user input
user_df = pd.DataFrame(user_input, index=[0])

# Preprocess the user input data
user_df_encoded = pd.get_dummies(user_df)
user_df_encoded = user_df_encoded.reindex(columns=X_encoded.columns, fill_value=0)

# Make predictions for the user input
prediction_probabilities = model.predict_proba(user_df_encoded)[:, 1]

# Display the prediction probabilities
for i, prob in enumerate(prediction_probabilities):
    print(f"The chance of depression for input {i+1}: {prob:.4f}")

# Plot the changing chances of depression
plt.plot(range(1, len(prediction_probabilities) + 1), prediction_probabilities, marker='o')
plt.xlabel('Input')
plt.ylabel('Chances of Depression')
plt.title('Changing Chances of Depression for User Inputs')
plt.xticks(range(1, len(prediction_probabilities) + 1))
plt.show()

# Make predictions for the user input
# ... Existing code ...

# Make predictions for the user input
# ... Existing code ...

# Make predictions for the user input
prediction = model.predict(user_df_encoded)

# Display the prediction
if prediction[0] == 1:
    print("Based on the provided information, you are predicted to have depression.")
else:
    print("Based on the provided information, you are predicted to not have depression.")

# Ask if the user wants to see the plot
show_plot = input("Do you want to see the plots? (Yes/No): ")

if show_plot.lower() == 'yes':
    # ... Existing plots ...

    # Perform predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = (y_pred == y_test).mean() * 100
    print("Model Accuracy: {:.2f}%".format(accuracy))

    # Create a confusion matrix
    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(confusion_matrix.columns))
    plt.xticks(tick_marks, confusion_matrix.columns)
    plt.yticks(tick_marks, confusion_matrix.index)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
