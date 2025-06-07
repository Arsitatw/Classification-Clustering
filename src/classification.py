from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

class GHIClassifier:
    def __init__(self, df, features, target):
        self.df = df
        self.features = features
        self.target = target
        self.model = None

    def train_random_forest(self):
        X = self.df[self.features]
        y = self.df[self.target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred)
        matrix = confusion_matrix(y_test, y_pred)

        return report, matrix
