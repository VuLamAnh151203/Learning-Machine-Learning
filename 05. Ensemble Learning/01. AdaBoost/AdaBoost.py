import numpy as np
import sklearn

# Specify the decision stump (Weak classifier)

class DecisionStump():
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.thresh = None
        self.alpha = None

    def predict(self,X):
        X_column = X[:, self.feature_idx]
        n_samples = X.shape[0]
        prediction = np.ones(n_samples)
        if self.polarity == 1:
            print(X_column < self.thresh)
            prediction[X_column < self.thresh] = -1
        else:
            prediction[X_column > self.thresh]  = -1
 
        return prediction 
    
class AdaBoost():
    def __init__(self, n_clf):
        self.n_clf = n_clf

    def fit(self, X,y):
        self.X = X
        self.y = y
        self.n_samples = X.shape[0]
        self.w = np.ones(self.n_samples) * (1/self.n_samples)
        n_features = X.shape[1]
        self.clf_list = []
        for clf in range(self.n_clf):
            # Create decision stump:
            clf = DecisionStump() 
            min_error = np.Inf
            for feature_idx in range(n_features):
                p = 1
                X_column = X[:, feature_idx]
                threshs = np.unique(X_column)
                for thresh in threshs:
                    prediction = np.ones(self.n_samples)
                    prediction[X_column < thresh] = -1

                    error = np.sum(self.w[self.y != prediction])

                    if error > 0.5:
                        p = -1
                        error = 1 - error

                    if min_error > error:
                        clf.polarity = p
                        clf.feature_idx = feature_idx
                        clf.thresh = thresh
                        min_error = error
                
        
            # Compute the "amount to say"
            EPS = 1e-10
            clf.alpha = 1/2 * np.log((1-error)/ (error + EPS)) 
            self.clf_list.append(clf)
            # update the weight so the next clf will focus on the hard example
            prediction = clf.predict(X)
            self.w *= np.exp(-clf.alpha * y * prediction)
            self.w /= np.sum(self.w)

    
    def predict(self,X):
        # for clf in self.clf_list:
        #     prediction = clf.predict(X)
        #     print(prediction)
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clf_list]
        clf_preds = np.sum(clf_preds, axis = 0)
        prediction = np.sign(clf_preds)

        return prediction
    



