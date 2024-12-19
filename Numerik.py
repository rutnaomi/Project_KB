import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('DatasetEmail/email.csv')

print(data.head())

texts = data['Message']
labels = data['Category']

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()

model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

new_emails = [
    "Congratulations! You've won a free ticket. Click here to claim.",
    "Hi, can we schedule a meeting for tomorrow?",
]
new_emails_tfidf = vectorizer.transform(new_emails)
new_predictions = model.predict(new_emails_tfidf)
print("\nPredictions for new emails:", new_predictions)
