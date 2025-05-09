from rule_classifier import RuleClassifier

# ************************ EXECUÇÃO ************************

train_path = "data/rapid_balanceado_treinamento.csv"
test_path = "data/rapid_balanceado_teste.csv"

# Model parameters
model_parameters = {
    'criterion': 'gini',
    'splitter': 'best',
    'max_leaf_nodes': None,
    'min_impurity_decrease': 0.0,
    'class_weight': None,
    'ccp_alpha': 0.0,
    'max_depth': 10,
    'min_samples_split': 2,
    'min_samples_leaf': 2,
    'max_features': None,
    'random_state': 42
}

# Generating the initial rule based model
classifier = RuleClassifier.new_classifier(train_path, test_path, model_parameters, algorithm_type='Decision Tree')

# Executing the rule analysis method
# remove_duplicates = "soft" (in the same tree, probably does not affect the final metrics), "hard" (between trees, may affect the final metrics), "custom" (custom function to remove duplicates) or "none" (no removal)
# remove_below_n_classifications = -1 (no removal), 0 (removal of rules with 0 classifications), or any other integer (removal of rules with equal or less than n classifications)
classifier.execute_rule_analysis(test_path, remove_duplicates="soft", remove_below_n_classifications=-1)

# Comparing initial and final results
classifier.compare_initial_final_results(test_path)