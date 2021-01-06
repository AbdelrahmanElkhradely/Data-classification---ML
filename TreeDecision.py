from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


class TreeDecision:
    def __init__(self,Xtrain, Xtest, Ytrain, Ytest):
        self.X_train=Xtrain
        self.y_train=Ytrain
        self.X_test=Xtest
        self.y_test=Ytest
        self.clf_entropy= self.train_using_entropy()
        self.y_pred=None


    # Function to perform training with entropy.
    def train_using_entropy(self):

        # Decision tree with entropy
        clf_entropy = DecisionTreeClassifier(
            criterion="entropy", random_state=None,
            max_depth=1000, min_samples_leaf=100)

        # Performing training
        clf_entropy.fit(self.X_train, self.y_train)
        return clf_entropy

    def prediction(self):
        # Predicton on test with giniIndex
        y_pred = self.clf_entropy.predict(self.X_test)
        print("Predicted values:")
        print(y_pred)
        self.y_pred=y_pred
        return y_pred

        # Function to calculate accuracy

    def cal_accuracy(self):
        print("Confusion Matrix: ",
              confusion_matrix(self.y_test, self.y_pred))

        print("Accuracy : ",
              accuracy_score(self.y_test, self.y_pred) * 100)

        print("Report:",
              classification_report(self.y_test, self.y_pred))

    def treedecision(self):
        self.prediction()
        self.cal_accuracy()
        plt.figure(figsize=(8, 8))
        self.clf_entropy.fit(self.X_train, self.y_train)
        probs = self.clf_entropy.predict_proba(self.X_test)
        fpr, tpr, thresholds = roc_curve(self.y_test, probs[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='%s (area = %0.2f)' % ("Tree decison", roc_auc))
        plt.plot([0, 1], [0, 1])
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Decison tree')
        plt.legend(loc="lower right")
        plt.show()