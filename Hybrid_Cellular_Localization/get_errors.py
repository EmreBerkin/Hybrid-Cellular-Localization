import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from filterpy.monte_carlo import systematic_resample
import data_loader
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

import hybrid_particle_filter as hpf
import hybrid_ukf_filter as hukf

class Get_Errors():

    def __init__(self):
        self.user_data = data_loader.user_data
        self.xm = self.user_data["x_m"].to_numpy()
        self.ym = self.user_data["y_m"].to_numpy()
        self.X = np.column_stack((self.xm, self.ym))
        self.Y = self.user_data[["RSRP", "RSRQ", "SINR"]]

    def getModel(self):

        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, 30],
            "min_samples_leaf": [1, 3, 5]
        }

        grid = GridSearchCV(
            RandomForestRegressor(),
            param_grid,
            cv=5,
            scoring="r2",
            n_jobs=-1
        )

        x_train, x_test, y_train, y_test = train_test_split(self.X, self.Y, train_size=0.7, random_state=42)
        model = grid.fit(x_train, y_train)
        return model
    
    def run_error_comparison(self):
        N = len(self.user_data)
        pf_error_list, ukf_error_list = np.zeros((100, N)), np.zeros((100, N))
        model = self.getModel()

        for i in range(100):
            print(i)

            #model = self.getModel()
            # Hybrid Particle Filter
            h_pf = hpf.Hybrid_Particle_Filter(model=model)
            est_list = h_pf.run_particle_filter()
            est_arr = np.array(est_list)
            x_hybrid_pf = est_arr[:, 0]
            y_hybrid_pf = est_arr[:, 1]

            # Hybrid UKF
            h_ukf = hukf.Hybrid_UKF_Filter(model=model)
            s_estimations_hukf = h_ukf.run_ukf()
            s_estimations_hukf = np.array(s_estimations_hukf)
            x_hybrid_ukf, y_hybrid_ukf = s_estimations_hukf[:,0], s_estimations_hukf[:,1]

            # Hybrid Particle Filter Error
            error_pf = np.sqrt((self.xm - x_hybrid_pf)**2 + (self.ym - y_hybrid_pf)**2)
            pf_error_list[i, :] = error_pf

            # Hybrid UKF Error
            error_ukf = np.sqrt((self.xm - x_hybrid_ukf)**2 + (self.ym - y_hybrid_ukf)**2)
            ukf_error_list[i, :] = error_ukf

        mean_error_pf, mean_error_ukf = np.mean(pf_error_list, axis=0), np.mean(ukf_error_list, axis=0)
        xx_pf = np.sort(mean_error_pf)            
        yy_pf = np.arange(1, N+1) / N  
        xx_ukf = np.sort(mean_error_ukf)            
        yy_ukf = np.arange(1, N+1) / N  

        plt.plot(xx_pf, yy_pf, lw=2, color="red")
        plt.plot(xx_ukf, yy_ukf, lw=2, color="green")
        plt.xlabel("Localization Error (m)")
        plt.ylabel("Probability")
        plt.grid(True)
        plt.legend(["Hybrid Particle Filter", "Hybrid UKF"])
        plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    get_error = Get_Errors()
    get_error.run_error_comparison()
