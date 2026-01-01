import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import joblib
import data_loader

# In this py doc we learn model for relevant cell ( For each user ).

class Cell_Model:
    def getUserInfo(self):
        self.cell_data = data_loader.cell_data.copy()
        self.x_m = self.cell_data["x_m"].to_numpy()
        self.y_m = self.cell_data["y_m"].to_numpy()

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

        x_train, x_test, y_train, y_test = train_test_split(
                X, Y, train_size=0.7, random_state=42
            )
        grid.fit(x_train, y_train)
        best_model = grid.best_estimator_

        tr_size = [0.7, 0.8, 0.9]
        r2_sc_list = []

        for tr_s in tr_size:
            x_train, x_test, y_train, y_test = train_test_split(
                X, Y, train_size=tr_s, random_state=42
            )
            best_model.fit(x_train, y_train)
            y_pred = best_model.predict(x_test)

            r2_sc = r2_score(y_test, y_pred, multioutput="variance_weighted")
            r2_sc_list.append(r2_sc)

            #print(f"Train size {tr_s} -> R2: {r2_sc:.4f}")

        joblib.dump(
            best_model,
            r"C:\Users\LENOVOGAMING\Desktop\Bitirme Düzenlenmiş Kodlar\Models\cell_best_model.pkl"
        )

        print("\nBest model parameters:")
        print(best_model)


if __name__ == "__main__":
    tum = Cell_Model()
    tum.getUserInfo()
    tum.getModelAndFit()
