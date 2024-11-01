import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle5 as pickle

def get_clean_data():

    # Load the data
    data = pd.read_csv("data/data.csv")
    
    # Drop unnecessary columns
    data = data.drop(columns=["Unnamed: 32", "id"], axis=1)
    
    # Map diagnosis to 0 and 1 
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data


def create_model(data):
    X = data.drop(columns=["diagnosis"], axis=1)
    Y = data["diagnosis"]

    # Standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Train the model
    model = LogisticRegression()
    model.fit(X_train, Y_train)

    # Evaluate the model
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    classificationreport= classification_report(Y_test, Y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:", classificationreport)

    return model, scaler





def main():
    data = get_clean_data()

    model, scaler = create_model(data)

    with open("model/model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

if __name__ == "__main__":
    main()