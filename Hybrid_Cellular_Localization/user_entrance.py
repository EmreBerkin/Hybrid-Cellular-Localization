import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import time

import hybrid_particle_filter as hpf
import hybrid_ukf_filter as hukf

class UserEntrance():

    def __init__(self):
        root = tk.Tk()
        root.title("Choose a target user")
        window_width = 550
        window_height = 450
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x_cordinate = int((screen_width/2) - (window_width/2))
        y_cordinate = int((screen_height/2) - (window_height/2))
        root.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")

        # Get Data
        self.data = pd.read_excel(r"C:\Users\LENOVOGAMING\Desktop\Bitirme Düzenlenmiş Kodlar\AllDataWithDistance.xlsx")
        cell_names = list(self.data["SERVING_CELL_NAME"].unique())
        
        # Get Cell
        cell = tk.Frame(root)
        cell.pack(fill="x", pady=20)
        cell_label = tk.Label(cell, text="Choose A Target Cell :", font=("Arial", 12))
        cell_label.pack(side="left", padx=10)

        # Combo Cells
        self.combo_cells = ttk.Combobox(
            cell,
            values=cell_names,
            state="readonly",
            width=20
        )

        self.combo_cells.set("Choose A Cell")
        self.combo_cells.pack(side="left")

        # Get Cluster
        cluster = tk.Frame(root)
        cluster.pack(fill="x", pady=20)
        cluster_label = tk.Label(cluster, text="Choose A Cluster :", font=("Arial", 12))
        cluster_label.pack(side="left", padx=10)

        # Combo Cluster
        self.combo_clusters = ttk.Combobox(
            cluster,
            values=[],
            state="readonly",
            width=20
        )

        self.combo_clusters.set("Choose A Cluster")
        self.combo_clusters.pack(side="left")

        # Choose Model Type
        model_type = tk.Frame(root)
        model_type.pack(fill="x", pady=20)
        model_type_label = tk.Label(model_type, text="Choose The Data Scope For Training :", font=("Arial", 12))
        model_type_label.pack(side="left", padx=10)

        # Combo Model type
        self.combo_models = ttk.Combobox(
            model_type,
            values=["Cell-specific training","Global training"],
            state="readonly",
            width=20
        )

        self.combo_models.set("Choose Model")
        self.combo_models.pack(side="left")

        # Choose Filter
        filter = tk.Frame(root)
        filter.pack(fill="x", pady=20)
        filter_label = tk.Label(filter, text="Choose A Filter :", font=("Arial", 12))
        filter_label.pack(side="left", padx=10)

        # Combo Filters
        self.combo_filter = ttk.Combobox(
            filter,
            values=["Hybrid Particle Filter", "Hybrid UKF"],
            state="readonly",
            width=20
        )

        self.combo_filter.set("Choose A Filter")
        self.combo_filter.pack(side="left")

        # Show Cell Button
        show_cell_button = tk.Button(cell, text="Show Cell", command=self.showCell)
        show_cell_button.pack(side="left", padx=10)

        # Estimation Button
        estimation_button = tk.Button(root, text="Start Estimation", command=self.estimateTrajectory)
        estimation_button.pack(pady=10)

        # Exit Button
        exit_button = tk.Button(root, text="Exit", command=root.destroy)
        exit_button.pack(pady=10)
        root.mainloop()

    def estimateTrajectory(self):
        selected_cell = self.combo_cells.get()
        self.selected_filter = self.combo_filter.get()
        selected_model = self.combo_models.get()

        if selected_model == "Cell-specific training":
            model_name = "cell_best_model"
        elif selected_model == "Global training":
            model_name = "target_user_best_model"

        cluster_data = self.getSelectedClusterData()
        x_true, y_true = cluster_data["x_m"], cluster_data["y_m"]
        
        if self.selected_filter == "Hybrid Particle Filter":
            h_pf = hpf.Hybrid_Particle_Filter(model_name=model_name)
            start = time.perf_counter()
            est_arr = np.array(h_pf.run_particle_filter())
            end = time.perf_counter()
            elapsed_time = end - start
            print(f"Filter : {self.selected_filter}, Model Name : {model_name}, Time Elapsed : {elapsed_time}")
            x_hybrid_pf = est_arr[:, 0]
            y_hybrid_pf = est_arr[:, 1]
            #messagebox.showinfo("Process Status","Trajectory estimation is currently in progress...")
            self.plotGraph(x_true, y_true, x_hybrid_pf, y_hybrid_pf)

        elif self.selected_filter == "Hybrid UKF":
            h_ukf = hukf.Hybrid_UKF_Filter(model_name=model_name)
            start = time.perf_counter()
            s_estimations_hukf = np.array(h_ukf.run_ukf())
            end = time.perf_counter()
            elapsed_time = end - start
            print(f"Filter : {self.selected_filter}, Model Name : {model_name}, Time Elapsed : {elapsed_time}")
            x_hybrid_ukf, y_hybrid_ukf = s_estimations_hukf[:,0], s_estimations_hukf[:,1]
            #messagebox.showinfo("Process Status","Trajectory estimation is currently in progress...")
            self.plotGraph(x_true, y_true, x_hybrid_ukf, y_hybrid_ukf)
    
    def getSelectedClusterData(self):
        selected_cell = self.combo_cells.get()
        cell_data = self.data[self.data["SERVING_CELL_NAME"] == selected_cell]

        X = np.column_stack((cell_data["x_m"], cell_data["y_m"]))
        dbscan = DBSCAN(eps=50, min_samples=10).fit(X)
        labels = dbscan.labels_

        selected_cluster_str = self.combo_clusters.get()
        if selected_cluster_str.startswith("Cluster "):
            selected_cluster = int(selected_cluster_str.split(" ")[1])
            mask = labels == selected_cluster
            cluster_data = cell_data[mask]
            return cluster_data
        else:
            return pd.DataFrame() 

    
    def showCell(self):
        selected_cell = self.combo_cells.get()
        cell_data = self.data[self.data["SERVING_CELL_NAME"] == selected_cell]
        x_true, y_true = cell_data["x_m"], cell_data["y_m"]

        X = np.column_stack((x_true, y_true))
        dbscan = DBSCAN(eps=50, min_samples=10).fit(X)
        labels = dbscan.labels_
        cluster_labels = sorted(list(set(labels) - {-1}))
        cluster_labels = [f"Cluster {label}" for label in cluster_labels]
        clusters = set(labels) - {-1}
        self.combo_clusters["values"] = cluster_labels

        plt.figure(figsize=(8,6))
        colors = plt.cm.tab10(np.linspace(0,1,len(clusters)))
        for k, col in zip(clusters, colors):
            mask = labels == k
            plt.scatter(X[mask,0], X[mask,1], color=col, label=f'Cluster {k}')

        plt.grid(True), plt.title(f"Cell Name : {selected_cell}"), plt.legend()
        plt.xlabel("$x_{m}$"), plt.ylabel("$y_{m}$"), plt.show()

    def plotGraph(self, x_measurements, y_measurements, x_estimations_m, y_estimations_m):
        plt.figure(figsize=(10,8))
        plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
        plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
        plt.scatter(x_measurements, y_measurements, color='blue', s=20, label='True Trajectory')
        plt.scatter(x_estimations_m, y_estimations_m, color='red', marker="s", s=20, label='Estimations')
        plt.plot(x_measurements, y_measurements, color='blue', linewidth=0.8, alpha=0.6)
        plt.plot(x_estimations_m, y_estimations_m, color='red', linewidth=0.8, alpha=0.6)
        plt.scatter(0, 0, color='m', marker="x", s=40, label='Cell Tower', zorder=5)
        plt.xlabel("X (m)"), plt.ylabel("Y (m)")
        plt.suptitle(f"True Trajectory vs {self.selected_filter} Estimations"), plt.legend()
        plt.grid(True), plt.show()


if __name__ == "__main__":
    UserEntrance()