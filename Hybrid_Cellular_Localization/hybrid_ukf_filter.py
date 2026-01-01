import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from filterpy.monte_carlo import systematic_resample
import data_loader
import joblib

class Hybrid_UKF_Filter():

    def __init__(self, model_name=None, model=None):
        self.user_data = data_loader.user_data.copy()
        self.user_data["TIME"] = pd.to_datetime(self.user_data["TIME"])
        self.dt_list = self.user_data["TIME"].diff().dt.total_seconds().to_numpy()
        self.dt_list[0] = 0.0 
        dt_min = 0.2  
        self.dt_list[self.dt_list < dt_min] = dt_min
        self.R = self.create_R_matrix()

        if model_name:
            # Get model whether it is learned for relevant cell or relevant target user
            self.model = joblib.load(fr"C:\Users\LENOVOGAMING\Desktop\Bitirme Düzenlenmiş Kodlar\Models\{model_name}.pkl")
        else:
            self.model = model

    def f_cv(self, sigma_points, dt):
        f_out = np.zeros_like(sigma_points)
        for i, sp in enumerate(sigma_points):
            x, y, vx, vy = sp
            f_out[i, 0] = x + vx * dt
            f_out[i, 1] = y + vy * dt
            f_out[i, 2] = vx
            f_out[i, 3] = vy
        return f_out

    def generate_sigma_points(self, s, n, S):
        sigma_points = [s]
        for i in range(n):
            sigma_points.append(s + S[:, i])
            sigma_points.append(s - S[:, i])
        return np.array(sigma_points)

    def generate_weights(self, lambda_s, n, alpha, beta):
        mean_weights = [lambda_s / (n + lambda_s)]
        cov_weights  = [lambda_s / (n + lambda_s) + (1 - alpha**2 + beta)]
        for _ in range(2 * n):
            mean_weights.append(1 / (2 * (n + lambda_s)))
            cov_weights.append(1 / (2 * (n + lambda_s)))
        return np.array(mean_weights), np.array(cov_weights)
    
    def compute_Q(self, dt, sigma_a):
        return sigma_a**2 * np.array([
            [dt**4/4,  0,        dt**3/2,  0],
            [0,        dt**4/4,  0,        dt**3/2],
            [dt**3/2,  0,        dt**2,     0],
            [0,        dt**3/2,  0,        dt**2]
        ])
    
    def create_R_matrix(self):
        std_RSRP = np.std(np.array(self.user_data["RSRP"]))
        std_RSRQ = np.std(np.array(self.user_data["RSRQ"]))
        std_SINR = np.std(np.array(self.user_data["SINR"]))

        R = np.diag([std_RSRP**2, std_RSRQ**2, std_SINR**2])
        return R

    def run_ukf(self):
        x_true = self.user_data["x_m"].to_numpy()
        y_true = self.user_data["y_m"].to_numpy()
        z_list = self.user_data[["RSRP", "RSRQ", "SINR"]].to_numpy()

        s = np.array([x_true[0], y_true[0], 3, 3], dtype=float)
        P = np.diag([50, 50, 5, 5])

        alpha, beta, kappa = 0.4, 2, 0
        n = 4
        lambda_s = alpha**2 * (n + kappa) - n
        sigma_a = 0.01

        s_estimations_hukf = []
        for i in range(len(x_true)):
            dt = self.dt_list[i]
            Q = self.compute_Q(dt, sigma_a)

            S = np.linalg.cholesky((n + lambda_s) * P)

            # Prediction
            sigma_points = self.generate_sigma_points(s, n, S)
            f_out = self.f_cv(sigma_points, dt)

            mean_w, cov_w = self.generate_weights(lambda_s, n, alpha, beta)
            s_mean = np.sum(mean_w[:, None] * f_out, axis=0)

            P_mean = Q + np.einsum(
                'i,ij,ik->jk',
                cov_w,
                f_out - s_mean,
                f_out - s_mean
            )

            # Update
            z_pred = self.model.predict(f_out[:, :2])
            z_mean = np.sum(mean_w[:, None] * z_pred, axis=0)

            P_z = self.R + np.einsum(
                'i,ij,ik->jk',
                cov_w,
                z_pred - z_mean,
                z_pred - z_mean
            )

            P_xz = np.einsum(
                'i,ij,ik->jk',
                cov_w,
                f_out - s_mean,
                z_pred - z_mean
            )

            K = P_xz @ np.linalg.inv(P_z)
            s = s_mean + K @ (z_list[i] - z_mean)
            P = P_mean - K @ P_z @ K.T

            s_estimations_hukf.append(s)

        return s_estimations_hukf

if __name__ == "__main__":

    model_name = "cell_best_model"
    h_ukf = Hybrid_UKF_Filter(model_name)
    s_estimations_hukf = h_ukf.run_ukf()

    user_data = data_loader.user_data
    xm = user_data["x_m"].to_numpy()
    ym = user_data["y_m"].to_numpy()

    s_estimations_hukf = np.array(s_estimations_hukf)
    x_hybrid_ukf, y_hybrid_ukf = s_estimations_hukf[:,0], s_estimations_hukf[:,1]

    min_x, max_x = min(xm.min(), x_hybrid_ukf.min()) - 50, max(xm.max(), x_hybrid_ukf.max()) + 50
    min_y, max_y = min(ym.min(), y_hybrid_ukf.min()) - 50, max(ym.max(), y_hybrid_ukf.max()) + 50

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


    plotGraph(xm, ym, x_hybrid_ukf, y_hybrid_ukf)