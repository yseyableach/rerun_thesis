import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from openpyxl import Workbook

def perform_classification(X_train, X_test, y_train, y_test):
  '''compare four algorith auc'''
  algorithms = {
      "Logistic Regression": LogisticRegression(random_state=42),
      "XGBoost": xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
      "LightGBM": lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
      "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
  }
  results = {}
  for algorithm_name, algorithm in algorithms.items():
      algorithm.fit(X_train, y_train)
      y_pred = algorithm.predict(X_test)
      report = classification_report(y_test, y_pred, target_names=iris.target_names, output_dict=True)
      results[algorithm_name] = report
  return results

def save_results_to_excel(results, excel_filename):
  '''save the result to xlsx'''
  workbook = Workbook()
  with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
      writer.book = workbook
      for algorithm_name, result in results.items():
          df = pd.DataFrame(result).reset_index()
          df.to_excel(writer, sheet_name=algorithm_name,index=False)
      workbook.save(excel_filename)
