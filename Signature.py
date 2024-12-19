import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_images(image_folder, img_size=(128, 128)):

    data = []
    labels = []
    classes = os.listdir(image_folder)
    for label, class_name in enumerate(classes):
        class_folder = os.path.join(image_folder, class_name)
        for img_file in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (150, 150))
            data.append(img.flatten())
            labels.append(label)
    return np.array(data), np.array(labels), classes


def split_data(features, labels, test_size=0.3, random_state=42):
    return train_test_split(features, labels, test_size=test_size, random_state=random_state)


def train_random_forest(X_train, y_train, n_estimators=100):

    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

def evaluate_model(model, X_test, y_test, class_names):

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")

    unique_labels = np.unique(np.concatenate((y_test, y_pred)))

    filtered_class_names = [class_names[i] for i in unique_labels]

    print(classification_report(y_test, y_pred, target_names=filtered_class_names)) # Use filtered_class_names

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=filtered_class_names, yticklabels=filtered_class_names) #Use filtered_class_names for confusion matrix labels
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
    return accuracy

def plot_accuracies(accuracies, model_names):

    plt.bar(model_names, accuracies, color=['blue', 'orange', 'green'])
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Comparison')
    plt.ylim(0, 1)
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f"{acc:.2f}", ha='center')
    plt.show()

if __name__ == "__main__":

    image_folder = "Dataset"

    print("Preprocessing images...")
    features, labels, class_names = preprocess_images(image_folder)

    print("Splitting data into training and validation sets...")
    X_train, X_test, y_train, y_test = split_data(features, labels)

    print("Training models...")
    rf_model = train_random_forest(X_train, y_train)

    print("Evaluating Random Forest model...")
    rf_accuracy = evaluate_model(rf_model, X_test, y_test, class_names)

    accuracies = [rf_accuracy]
    model_names = ['rf']
    plot_accuracies(accuracies, model_names)