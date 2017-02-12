from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

scores = ['precision', 'recall']

# Set the parameters by cross-validation
tuned_parameters = [{'loss':["hinge","modified_huber","squared_hinge"],'alpha': [0.00001,0.0001,0.001,0.01],"penalty":["l1","l2","elasticnet"]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SGDClassifier(shuffle=True,fit_intercept=False,n_jobs=-1,learning_rate="optimal",penalty="l2",class_weight="balanced",n_iter=5), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
