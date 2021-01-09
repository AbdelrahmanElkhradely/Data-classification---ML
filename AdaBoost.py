from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt



class AdaBoost:
    def __init__(self, Xtrain, Xtest, Ytrain, Ytest):
        self.X_train = Xtrain
        self.y_train = Ytrain
        self.X_test = Xtest
        self.y_test = Ytest
        self.y_pred = None
        self.classfier=self.train()



    def train(self):
        # Create a Gaussian Classifier
        gnb = AdaBoostClassifier(n_estimators=100, random_state=None)
        # Train the model using the training sets
        gnb.fit(self.X_train, self.y_train)
        # Predict the response for test dataset
        self.y_pred = gnb.predict(self.X_test)
        print("Predicted values:")
        print(self.y_pred)
        return gnb

    def cal_accuracy(self):
        print("Confusion Matrix: ",
              confusion_matrix(self.y_test, self.y_pred))

        print("Accuracy : ",
              accuracy_score(self.y_test, self.y_pred) * 100)

        print("Report:",
              classification_report(self.y_test, self.y_pred))
    def adaboost(self):
        self.cal_accuracy()
        plt.figure(figsize=(8, 8))
        self.classfier.fit(self.X_train, self.y_train)
        probs = self.classfier.predict_proba(self.X_test)
        fpr, tpr, thresholds = roc_curve(self.y_test, probs[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='%s (area = %0.2f)' % ("ada boost", roc_auc))
        plt.plot([0, 1], [0, 1])
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Ada Boost')
        plt.legend(loc="lower right")
        plt.show()