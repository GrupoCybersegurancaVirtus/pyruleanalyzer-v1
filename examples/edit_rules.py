from pyruleanalyzer.rule_classifier import RuleClassifier

# Path to the test data
test_path = "examples/data/test.csv"

# Loading the existing model
classifier = RuleClassifier.load("examples/files/final_model.pkl")

# Loading the existing model
# classifier = RuleClassifier.load("examples/files/edited_model.pkl")

# Comparing initial and final results
classifier.compare_initial_final_results(test_path)

# Editing the rules
classifier.edit_rules()