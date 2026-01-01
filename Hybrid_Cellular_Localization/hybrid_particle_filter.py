import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from filterpy.monte_carlo import systematic_resample
import data_loader
import joblib


class Hybrid_Particle_Filter():

    def __init__(self, model_name=None, model=None):
        self.user_data = data_loader.user_data.copy()

        self.user_data["TIME"] = pd.to_datetime(self.user_data["TIME"])
        self.dt_list = self.user_data["TIME"].diff().dt.total_seconds().to_numpy()
        self.dt_list[0] = 0.0

        self.R = self.create_R_matrix()

        if model_name:
            # Get model whether it is learned for relevant cell or relevant target user
            self.model = joblib.load(fr"C:\Users\LENOVOGAMING\Desktop\Bitirme Düzenlenmiş Kodlar\Models\{model_name}.pkl")
        else:
            self.model = model

    def create_R_matrix(self):
        std_RSRP = np.std(np.array(self.user_data["RSRP"]))
        std_RSRQ = np.std(np.array(self.user_data["RSRQ"]))
        std_SINR = np.std(np.array(self.user_data["SINR"]))

        R = np.diag([std_RSRP**2, std_RSRQ**2, std_SINR**2])
        return R

    def calculate_Q_matrix(self, dt, sigma_w=0.01):
        dt2 = dt**2
        dt3 = dt**3
        dt4 = dt**4

        Q = sigma_w ** 2 * np.array([
            [dt4/4, 0,      dt3/2, 0,     dt2/2, 0    ],
            [0,     dt4/4,  0,     dt3/2, 0,     dt2/2],
            [dt3/2, 0,      dt2,   0,     dt,    0    ],
            [0,     dt3/2,  0,     dt2,   0,     dt   ],
            [dt2/2, 0,      dt,    0,     1,     0    ],
            [0,     dt2/2,  0,     dt,    0,     1    ]
            ])
        
        return Q

    def new_x_particle(self, x_particle, dt):
        x, y, Vx, Vy, ax, ay = x_particle

        x_new = x + Vx * dt + 0.5 * ax * dt**2
        y_new = y + Vy * dt + 0.5 * ay * dt**2
        Vx_new = Vx + ax * dt
        Vy_new = Vy + ay * dt

        return np.array([x_new, y_new, Vx_new, Vy_new, ax, ay])
    
    def run_particle_filter(self, pf_size = None):
        if pf_size:
            N = pf_size
        else:
            N = 1000
            
        data = self.user_data
        T = len(data)

        data = data.iloc[:T]
        x0 = data["x_m"].iloc[0]
        y0 = data["y_m"].iloc[0]

        est_list = []

        z_list = data[["RSRP", "RSRQ", "SINR"]].to_numpy()
        P0 = np.diag([20, 20, 3, 3, 0, 0])

        m0 = np.array([x0, y0, 4, 4, 0, 0])
        parts = np.random.multivariate_normal(mean=m0, cov=P0, size=N)
        w = np.ones(N) / N

        Rin = np.linalg.inv(self.R)

        for t in range(T):

            z = z_list[t].reshape(1, 3)
            dt = self.dt_list[t]
            Qm = self.calculate_Q_matrix(dt, sigma_w=0.01)

            proc = np.random.multivariate_normal(
                np.zeros(6), Qm + 1e-6*np.eye(6), size=N
            )

            parts = np.array([self.new_x_particle(p, dt) for p in parts])
            parts = parts + proc

            pos = parts[:, :2]

            zhat = self.model.predict(pos).reshape(N, 3)

            diff = zhat - z
            exponent = -0.5 * np.sum((diff @ Rin) * diff, axis=1)
            like = np.exp(exponent) + 1e-300

            w *= like
            w /= np.sum(w)

            est = np.sum(parts * w[:, None], axis=0)
            est_list.append(est)

            Neff = 1 / np.sum(w**2)
            if Neff < N / 2:
                idx = systematic_resample(w)
                parts = parts[idx]
                w = np.ones(N) / N

        return est_list
    
if __name__ == "__main__":

    model_name = "cell_best_model"
    hpf = Hybrid_Particle_Filter(model_name)
    est_list = hpf.run_particle_filter()

    user_data = data_loader.user_data
    xm = user_data["x_m"].to_numpy()
    ym = user_data["y_m"].to_numpy()

    est_arr = np.array(est_list)
    x_hybrid_pf = est_arr[:, 0]
    y_hybrid_pf = est_arr[:, 1]

    min_x, max_x = min(xm.min(), x_hybrid_pf.min()) - 50, max(xm.max(), x_hybrid_pf.max()) + 50
    min_y, max_y = min(ym.min(), y_hybrid_pf.min()) - 50, max(ym.max(), y_hybrid_pf.max()) + 50

    def plotGraph(x_measurements, y_measurements, x_estimations_m, y_estimations_m):
        plt.figure(figsize=(10,8))
        plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
        plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
        plt.scatter(x_measurements, y_measurements, color='blue', s=20, label='True Trajectory')
        plt.scatter(x_estimations_m, y_estimations_m, color='red', marker="s", s=20, label='Estimations')
        plt.plot(x_measurements, y_measurements, color='blue', linewidth=0.8, alpha=0.6)
        plt.plot(x_estimations_m, y_estimations_m, color='red', linewidth=0.8, alpha=0.6)
        plt.scatter(0, 0, color='m', marker="x", s=40, label='Cell Tower', zorder=5)
        plt.xlabel("X (m)"), plt.ylabel("Y (m)")
        plt.suptitle("True Trajectory vs Hybrid UKF Estimations"), plt.legend()
        plt.xlim(min_x, max_x), plt.ylim(min_y, max_y)
        plt.grid(True), plt.show()


    plotGraph(xm, ym, x_hybrid_pf, y_hybrid_pf)