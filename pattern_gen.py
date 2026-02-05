import random
import numpy as np
import matplotlib.pyplot as plt

def generate_pattern(seed, outfile, n=10, m=5):
    random.seed(seed)
    np.random.seed(seed)
    # n = np.random.randint(10, 20)
    # n = 3
    print(n)
    # dx = 1/m
    dx = 1.4/m
    crack_l = 0.25
    # x_interval = np.array([[dx*i, dx*(i+1)] for i in range(m)]) *2
    x_interval = np.array([[dx*i, dx*(i+1)] for i in range(m)]) + 0.3
    # y_interval = np.array([[dx*i, dx*(i+1)] for i in range(m)]) *2
    y_interval = np.array([[dx*i, dx*(i+1)] for i in range(m)]) + 0.3
    # create a list of all 2d intervals
    intervals = []
    for i in range(m):
        for j in range(m):
            intervals.append([x_interval[i], y_interval[j]])
    ind = random.sample(range(len(intervals)), n)
    intervals = [intervals[i] for i in ind]
    # Interval centers (n, 2) (x, y)
    interval_centers = np.array([[(x[0]+x[1])/2, (y[0]+y[1])/2] for x, y in intervals])
    # Initial crack vector (n, 2, 2) [(x1, y1), (x2, y2)]
    initial_crack_vector = np.array([[[interval_centers[i][0]-crack_l/2, interval_centers[i][1]], [interval_centers[i][0]+crack_l/2, interval_centers[i][1]]] for i in range(interval_centers.shape[0])])
    initial_crack_vector_center = np.mean(initial_crack_vector, axis=1)
    init_crack_vector_t = [initial_crack_vector[i] - initial_crack_vector_center[i] for i in range(n)]
    theta = np.random.rand(n)*np.pi
    t = np.random.rand(n, 2)*(1/m/4)
    transformed_cracks = []
    for i in range(n):
        R_mat = np.array([[np.cos(theta[i]), -np.sin(theta[i])], [np.sin(theta[i]), np.cos(theta[i])]])
        transformed_cracks.append(np.dot(R_mat, init_crack_vector_t[i].T).T + initial_crack_vector_center[i] +t[i])
    transformed_cracks = np.array(transformed_cracks)
    np.save(outfile, transformed_cracks)
    # plot the cracks in a 2x2 domain
    fig, ax = plt.subplots()
    for i in range(n):
        x, y = transformed_cracks[i].T
        ax.plot(x, y, 'r')
    ax.set_xlim([0, 2])
    ax.set_ylim([0, 2])
    ax.set_aspect('equal')
    plt.savefig(outfile.replace('.npy', '.png'))
    plt.close()

if __name__ == '__main__':
    import os
    out_file = 'pfm_dataset/initial_cracks_1c'
    os.makedirs(out_file, exist_ok=True)
    for i in range(1000):
        seed_value = random.randint(1, 1000000000)
        # seed_value = 1726
        outfile_npy = f"{out_file}/{seed_value}.npy"
        generate_pattern(seed_value, outfile_npy, n=1, m=5)