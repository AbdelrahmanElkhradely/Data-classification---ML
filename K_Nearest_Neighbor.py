from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class K_Nearest_Neighbor:
    def __init__(self,Xtrain, Xtest, Ytrain, Ytest):
        self.X_train = Xtrain
        self.y_train = Ytrain
        self.X_test = Xtest
        self.y_test = Ytest
        self.y_pred = None
        self.classfier = self.train()

    def train(self):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(self.X_train)
        X_train = scaler.transform(self.X_train)
        X_test = scaler.transform(self.X_test)
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors=100)
        classifier.fit(X_train, self.y_train)
        self.y_pred = classifier.predict(X_test)
        print("Predicted values:")
        print(self.y_pred)
        return classifier

    def cal_accuracy(self):
        print("Confusion Matrix: ",
              confusion_matrix(self.y_test, self.y_pred))

        print("Accuracy : ",
              accuracy_score(self.y_test, self.y_pred) * 100)

        print("Report:",
              classification_report(self.y_test, self.y_pred))
    def knearest (self):

        self.cal_accuracy()
        plt.figure(figsize=(8, 8))
        self.classfier.fit(self.X_train, self.y_train)
        probs = self.classfier.predict_proba(self.X_test)
        fpr, tpr, thresholds = roc_curve(self.y_test, probs[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='%s (area = %0.2f)' % ("naive bayes", roc_auc))
        plt.plot([0, 1], [0, 1])
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - naive bayes')
        plt.legend(loc="lower right")
        plt.show()
