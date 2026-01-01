import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import joblib
import data_loader

# In this py doc we learn model just for relevant user.

class TargetUserModel:
    def getUserInfo(self):
        cell_data = data_loader.cell_data
        self.cell_data = cell_data[cell_data["y_m"] > -700]
        self.x_m, self.y_m = self.cell_data["x_m"], self.cell_data["y_m"]

    def getModelAndFit(self):
        X = np.column_stack((self.x_m, self.y_m))
        Y = self.cell_data[["RSRP", "RSRQ", "SINR"]]

        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, 30],
            "min_samples_leaf": [1, 3, 5]
        }

        # For parameter optimization
        grid = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid,
            cv=5,
            scoring="r2",
            n_jobs=-1
        )

        # For which train size effects r2 score (Overfitting Test)
        tr_size = [0.7, 0.8, 0.9]
        r2_sc_list = []
        r2_model = dict()
        for tr_s in tr_size:
            x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=tr_s, random_state=42)
            learned_model = grid.fit(x_train, y_train)
            y_pred = learned_model.predict(x_test)
            r2_sc = r2_score(y_test, y_pred, multioutput="uniform_average")
            r2_sc_list.append(r2_sc)
            r2_model[tr_s] = learned_model.best_estimator_
        
        max_r2_index = np.argmax(r2_sc_list)
        best_model = r2_model[tr_size[max_r2_index]]
        
        joblib.dump(best_model, r"C:\Users\LENOVOGAMING\Desktop\Bitirme Düzenlenmiş Kodlar\Models\target_user_best_model.pkl") # Save best model 
        print(best_model)

if __name__ == "__main__":
    tum = TargetUserModel()
    tum.getUserInfo()
    tum.getModelAndFit()
