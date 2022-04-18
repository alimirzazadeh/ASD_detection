import joblib
boost_model = joblib.load('analysis/gradient_boost.pkl')

forest_model = joblib.load('analysis/forest.pkl')

tree_model = joblib.load('analysis/decision_tree.pkl')

random_patient = [[1,0,0,0,0,0,0,1,1,0,1,28,0,0,1,0,0]]
print(boost_model.predict(random_patient))
print(forest_model.predict(random_patient))
print(tree_model.predict(random_patient))