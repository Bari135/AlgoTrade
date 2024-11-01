import xgboost as xgb
import pandas as pd
import numpy as np
import joblib  # For saving and loading the model
import os
import datetime as dt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from procces_data import prepare_data
from sklearn.utils.class_weight import compute_class_weight

class XGBoostModel:
    def __init__(self, symbol, test_size=0.33, random_state=42, num_boost_round=500, early_stopping_rounds=20, model_path='xgboost_model.pkl', data_path='xgboost_data.npz'):
        self.symbol = symbol
        self.test_size = test_size
        self.random_state = random_state
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
        self.model_path = model_path
        self.data_path = data_path
        self.feature_names_path = 'feature_names.txt'
        self.label_encoder = LabelEncoder()
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None


    def prepare_data(self):
        data = prepare_data(self.symbol)
        return data
    


    def train_model(self):
        if not os.path.exists(self.model_path) or not os.path.exists(self.data_path):
            data = self.prepare_data()

            #filter the data to presereve some for the backtesting
            data['open_time'] = pd.to_datetime(data['open_time'])
            data = data[data['open_time'] < pd.to_datetime('2024-01-01')].copy()


            # Separate features and target
            x = data.drop(columns=['next_open','real_signal', 'price_change', 'close', 'open_time'], axis=1)  # Features
            y = data['real_signal']  # Target variable

            with open(self.feature_names_path, 'w') as f:
                for feature in x.columns:
                    f.write(f"{feature}\n")

            # Split the data into training and testing sets
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, shuffle=True, test_size=self.test_size, random_state=self.random_state, stratify=y)

            # Save the train/test data
            np.savez(self.data_path, X_train=self.X_train, X_test=self.X_test, y_train=self.y_train, y_test=self.y_test)
            

            param_grid = {
                'booster': ['gbtree'],
                'objective': ['multi:softmax'],
                'num_class': [3],  # Number of classes
                'eta': [0.01, 0.05, 0.1, 0.2],  # Learning rate
                'max_depth': [4, 6, 8],  # Max depth of a tree
                'eval_metric': ['mlogloss']
            }

            # Initialize the XGBoost model
            xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

           
            # Perform grid search
            grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1)
            grid_search.fit(self.X_train, self.y_train)
            
            
            # Save the best model
            self.model = grid_search.best_estimator_
            joblib.dump(self.model, self.model_path)
        else:
            data = np.load(self.data_path)
            self.X_train = data['X_train']
            self.X_test = data['X_test']
            self.y_train = data['y_train']
            self.y_test = data['y_test']
            
        
    def evaluate_model(self):
        if self.model is None:
            if not os.path.exists(self.model_path):
                raise ValueError("Model has not been trained yet. Please train the model first.")
            self.model = joblib.load(self.model_path)
        
        if self.X_test is None or self.y_test is None:
            if not os.path.exists(self.data_path):
                raise ValueError("Train/test data not found. Please train the model first.")
            data = np.load(self.data_path)    
            self.X_test = data['X_test']
            self.y_test = data['y_test']
            
        
        with open(self.feature_names_path, 'r') as f:
            feature_names = f.read().splitlines()
        
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)

        # Evaluate the model
        accuracy = accuracy_score(self.y_test, y_pred)
        conf_matrix = confusion_matrix(self.y_test, y_pred)

        print(f"finished training, Accuracy: {accuracy}")
        print("Confusion Matrix:")
        print(conf_matrix)

        # Plot feature importance
        xgb.plot_importance(self.model)
        plt.show()


    def predict(self, data):
        if self.model is None:
            if not os.path.exists(self.model_path):
                raise ValueError("Model has not been trained yet. Please train the model first.")
            self.model = joblib.load(self.model_path)

        x = data.drop(columns=['next_open', 'real_signal', 'price_change', 'close', 'position', 'qty', 'balance', 'open_time'], axis=1)

        predictions = self.model.predict(x)
        return predictions
    


# if __name__ == "__main__":
#     model = XGBoostModel(symbol='BTC')
#     model.train_model()
#     model.evaluate_model()
