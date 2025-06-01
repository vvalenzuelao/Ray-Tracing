import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 18
})

def Inicialization(electron_quantity):
    x_i = []
    y_i = np.random.normal(0, 0.02, electron_quantity)
    y_i = np.clip(y_i, -0.02, 0.02)
    for y in y_i:
        x_i.append(-0.1)

    Energies_D = np.random.normal(0.7, 0.7, electron_quantity)
    Energies_D = np.clip(Energies_D, 0, 1.7)

    return x_i, y_i, Energies_D

def ElectronEntrancePoint_Camera(xi, yi, Energies, length):
    def Angle_Disperssion(r, d):
        q = 1.602e-19
        I = 1200
        k_MeV = 0.7
        k = k_MeV * 1.60218e-13
        m = 9.109e-31
        c = 3.0e8
        R_H = 0.01
        epsilon_0 = 8.854e-12

        gamma = 1 + k / (m * c**2)
        v = c * np.sqrt(1 - (1 / gamma)**2)
        A = np.pi * R_H**2
        tan_phi = (q * I * d * r) / (4 * v**3 * A * m * epsilon_0 * gamma)
        return tan_phi

    Electron_entrances = []

    def line(slope, px, py, x):
        return py + slope * (x - px)

    for i in range(len(xi)):
        point_x, point_y = xi[i], yi[i]
        Energy = Energies[i]

        tan_phi = Angle_Disperssion(point_y, 0.1)
        
        Entrance_position_y = line(tan_phi, point_x, point_y, 0.02)
        if -0.0015 <= Entrance_position_y <= 0.0015:
            tan_entrance_angle = Angle_Disperssion(point_y, 0.125)
            Entrance_to_B = line(tan_entrance_angle, point_x, point_y, length)
            phi = np.arctan(Entrance_to_B)
            point_int = (length, Entrance_to_B)
            Electron_entrances.append((point_int, Energy, phi))

    return Electron_entrances

def ElectronExitPoint(electron_entrances, B_field_strength, yf):
    Electron_Exits = []

    q = 1.602e-19
    m = 9.109e-31
    c = 3e8

    for intersection in electron_entrances:
        (x0, y0), E, alpha = intersection

        E_J = E * 1.602e-13
        gamma = 1 + E_J / (m * c**2)
        v = c * np.sqrt(1 - (1 / gamma)**2)
        r = (gamma * m * v) / (q * B_field_strength)

        if r >= yf:
            theta = np.arccos(np.cos(alpha) - (yf - y0) / r)
            xf = x0 + r * np.sin(theta) - r * np.sin(alpha)
            dir_xf = np.cos(theta)
            dir_yf = np.sin(theta)
            if xf <= 0.125:
                Electron_Exits.append(((x0, y0), (xf, yf), (dir_xf, dir_yf), E, r, theta, alpha))
            else:
                xfv = 0.125
                yfv = y0 - r * np.cos(np.arcsin(np.sin(alpha) + (xfv-x0)/r)) - r * np.cos(alpha)
                theta = np.arcsin(np.sin(alpha) + (xfv-x0) / r)
                dir_xf = np.cos(theta)
                dir_yf = np.sin(theta)
                Electron_Exits.append(((x0, y0), (xfv, yfv), (dir_xf, dir_yf), E, r, theta, alpha))
    return Electron_Exits

def Intersections(electron_exits, yff):
    Electron_Intersections = []

    for electron in electron_exits:
        (x0, y0), (xf, yf), (dir_xf, dir_yf), E, r, theta, alpha = electron
        if xf <= 0.125:
            xff = xf + 1 / np.tan(theta) * (yff - yf)
            Electron_Intersections.append(((x0, y0), (xf, yf), (xff, yff), E, r, theta, alpha))
        else:
            xffv = 0.18
            yffv = yf + (xffv-xf) * np.tan(theta)
            Electron_Intersections.append(((x0, y0), (xf, yf), (xffv, yffv), E, r, theta, alpha))

    return Electron_Intersections

def compute_trajectory(start_point, r, alpha, theta, yff, num_points=100):
    x0, y0 = start_point
    angles = np.linspace(alpha, alpha + theta, num_points)
    x_arc = x0 + r * (np.sin(angles) - np.sin(alpha))
    y_arc = y0 - r * (np.cos(angles) - np.cos(alpha))

    xf = x_arc[-1]
    yf = y_arc[-1]

    x_cutoff = 0.18

    # Calculamos intersección con yff
    xff = xf + (yff - yf) / np.tan(theta)
    if xff > x_cutoff:
        xff = x_cutoff
        yff = yf + (xff - xf) * np.tan(theta)
    x_line = np.array([xf, xff])
    y_line = np.array([yf, yff])

    # Recorta si algún valor del arco supera 0.18
    arc_mask = x_arc <= x_cutoff
    x_arc = x_arc[arc_mask]
    y_arc = y_arc[arc_mask]

    # Junta arco y línea
    x_total = np.concatenate((x_arc, x_line))
    y_total = np.concatenate((y_arc, y_line))

    # Recorta si algún valor de línea supera 0.18
    line_mask = x_total <= x_cutoff
    x_total = x_total[line_mask]
    y_total = y_total[line_mask]

    return x_total, y_total


electron_quantity = 1000000
B_field_strength = 0.07
x_entrada = 0.025
yf = 0.02
yff = 0.05

x_i, y_i, Energies_D = Inicialization(electron_quantity)
electron_entances = ElectronEntrancePoint_Camera(x_i, y_i, Energies_D, x_entrada)
electron_exits = ElectronExitPoint(electron_entances, B_field_strength, yf)
electron_intersections = Intersections(electron_exits, yff)

# Trayectorias de electrones
all_energies = [e[-4] for e in electron_intersections]
norm = colors.Normalize(vmin=min(all_energies), vmax=max(all_energies))
cmap = cm.plasma

for electron in electron_intersections:
    (x0, y0), (xf, yf), (xff, yff), E, r, theta, alpha = electron
    x_vals, y_vals = compute_trajectory((x0, y0), r, alpha, theta, yff)
    color = cmap(norm(E))
    plt.plot(x_vals, y_vals, color=color, alpha=0.5)

# Línea horizontal de y = 0.02 desde x = 0.025 hasta x = 0.125
plt.plot([0.025, 0.125], [0.02, 0.02], color='green', linestyle='--', label='Fin del campo magnético')
# Línea vertical en x = 0.125 desde y = 0 hasta y = 0.02
plt.vlines(x=0.125, ymin=0, ymax=0.02, color='green', linestyle='--')

# Línea horizontal de y = 0.05 desde x = 0.125 hasta x = 0.18
plt.plot([0.03, 0.17], [0.05, 0.05], color='orange', linestyle='--', label='Image Plate')
# Línea vertical en x = 0.18 desde y = 0 hasta y = 0.05
#plt.vlines(x=0.18, ymin=0, ymax=0.0, color='orange', linestyle='--')

plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Trayectorias de electrones')
plt.legend()
plt.grid(True)

# Colorbar de energía
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.set_label('Energía (MeV)')

plt.tight_layout()
plt.show()
#plt.savefig('Trayectorias', dpi = 300)


energies_of_interest = np.arange(0.4, 1.7, 0.1)
mean_x = []
min_x = []
max_x = []

for E_target in energies_of_interest:
    x_positions = [e[2][0] for e in electron_intersections if np.abs(e[3] - E_target) < 0.05]
    if x_positions:
        mean_x.append(np.mean(x_positions))
        min_x.append(np.min(x_positions))
        max_x.append(np.max(x_positions))
    else:
        mean_x.append(np.nan)
        min_x.append(np.nan)
        max_x.append(np.nan)

lower_errors = [mean - min_ for mean, min_ in zip(mean_x, min_x)]
upper_errors = [max_ - mean for mean, max_ in zip(mean_x, max_x)]
asymmetric_errors = [lower_errors, upper_errors]

plt.figure(figsize=(10, 6))
plt.errorbar(energies_of_interest, mean_x, yerr=asymmetric_errors, fmt='o', capsize=5, color='blue', ecolor='gray')
plt.xlabel('Energía (MeV)')
plt.ylabel('Posición en x (m) cuando y = 0.05')
plt.title('Posición en x vs Energía')
plt.grid(True)
plt.tight_layout()
plt.show()
#plt.savefig('Precision', dpi = 300)