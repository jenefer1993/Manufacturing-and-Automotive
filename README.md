# Manufacturing-and-Automotive
Contextual Predictive Maintenance (IoT Edge AI)
Executive Problem Statement
Predictive maintenance offers the potential to drastically reduce unexpected breakdowns and
lower fleet or factory maintenance costs. However, most existing ML systems rely exclusively on
internal diagnostic signals and fail in real-world deployment. A machine's failure is rarely
isolated; it is heavily influenced by external factors such as weather conditions, factory load, or
traffic density.
The objective of this project is to build an advanced contextual data fusion framework. The
intern will integrate internal IoT telemetry (sensor data like vibration, current, and temperature)
with external environmental signals acquired via third-party APIs or Vehicle-to-Everything (V2X)
logs. The goal is to accurately predict mechanical failures before they happen using robust
# ensemble modeling.

# Macro F1 Score

The F1 Score is a metric that combines Precision and Recall into a single value.

  F1 = (Precision+Recall)/(2×Precision×Recall)

For multi-class classification, there are different ways to calculate the overall F1 score. One common method is Macro F1 Score.

What is Macro F1?
Calculate the F1 score for each class separately.
Take the simple average of all class F1 scores.
Macro F1 = (F11​+F12+⋯+F1n)/n
  where n is the number of classes.

Why use Macro F1?
Gives equal importance to every class.
Useful when you want to know how well the model performs across all classes, including minority classes.
Can reveal poor performance on rare classes that overall accuracy might hide.

In imbalanced datasets, Macro F1 is often preferred because it does not let large classes dominate the evaluation.

# GOAL
1) Optimize the Macro F1 score
2) Maintains high predictive accuracy (Macro F1 ≥ 0.85)

# Who Uses?
the Fleet/Plant Manager focuses on "WHAT will fail and WHEN", while the Reliability Engineer focuses on "WHY it will fail."

# Minimum Viable Product Specifications 
Analyzes the feature importance charts 
(e.g., SHAP values) to see if external 
temperature or internal vibration is 
driving the anomaly. 
The foundational feature is the Contextual Data Fusion Pipeline. The intern must merge 
high-frequency IoT sensor time-series data with external API data (like weather or load 
conditions) using complex Pandas windowing operations. 
The core deliverable is the Classification Pipeline. The intern will utilize advanced gradient 
boosting algorithms (like LightGBM) combined with strict Synthetic Minority Over-sampling 
Technique (SMOTE) applied only within training folds during cross-validation to prevent data 
leakage. 
