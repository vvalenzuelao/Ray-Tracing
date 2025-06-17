import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.colors as colors
from collections import defaultdict


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
    x_i = np.full(electron_quantity, -0.1)
    y_i = np.random.normal(0, 0.02, electron_quantity)
    y_i = np.clip(y_i, -0.02, 0.02)
    z_i = np.random.normal(0, 0.02, electron_quantity)
    z_i = np.clip(z_i, -0.02, 0.02)
    Energies_D = np.random.normal(0.7, 0.3, electron_quantity)
    Energies_D = np.clip(Energies_D, 0.2, 1.6)
    return x_i, y_i, z_i, Energies_D

def ElectronEntrancePoint_Camera(xi, yi, zi, Energies, length):
    def Radial_Dispersion(r, d):
        q = 1.602e-19
        I = 1200
        k = 0.7 * 1.60218e-13
        m = 9.109e-31
        c = 3e8
        R_H = 0.01
        epsilon_0 = 8.854e-12
        gamma = 1 + k / (m * c**2)
        v = c * np.sqrt(1 - (1 / gamma)**2)
        A = np.pi * R_H**2
        tan_phi = (q * I * d * r) / (4 * v**3 * A * m * epsilon_0 * gamma)
        return tan_phi

    Electron_entrances = []

    for i in range(len(xi)):
        r_perp = np.sqrt(yi[i]**2 + zi[i]**2)
        if r_perp != 0:
            tan_phi_y = Radial_Dispersion(yi[i], length)
            delta_r_y = tan_phi_y * (0.02 - xi[i])
            y_ent = yi[i] + delta_r_y

            tan_phi_z = Radial_Dispersion(zi[i], length)
            delta_r_z = tan_phi_z * (0.02 - xi[i])
            z_ent = zi[i] + delta_r_z

            unit_vector = 0
            if np.sqrt(y_ent**2 + z_ent**2) <= 0.0015:

                tan_phi_y2 = Radial_Dispersion(yi[i], length + 0.025)
                delta_r_y2 = tan_phi_y2 * (0.025 - xi[i])
                y_proj = yi[i] + delta_r_y2

                tan_phi_z2 = Radial_Dispersion(zi[i], length + 0.025)
                delta_r_z2 = tan_phi_z2 * (0.025 - xi[i])
                z_proj = zi[i] + delta_r_z2
                phi = 0
                #norm_vec = np.sqrt(1**2 + tan_phi_y2**2 + tan_phi_z2**2)
                #unit_vector_3D = np.array([1, tan_phi_y2, tan_phi_z2]) / norm_vec

                v_vec = np.array([0.025 - xi[i], y_proj - yi[i], z_proj - zi[i]])
                norm_v = np.linalg.norm(v_vec)
                unit_vector_3D = v_vec / norm_v

                vec_entrada_xy = np.array([0.025 - xi[i], y_proj - yi[i]])
                beta = np.arctan(tan_phi_y2)#np.arctan2(vec_entrada_xy[1], vec_entrada_xy[0])

                Electron_entrances.append(((0.025, y_proj, z_proj), Energies[i], phi, beta, unit_vector, unit_vector_3D))
    return Electron_entrances

def ElectronExitPoint(Electron_entrances, B, yf, E_z):
    Electron_Exits = []
    q = 1.602e-19      # Carga del electrón
    m = 9.109e-31      # Masa del electrón
    c = 3e8            # Velocidad de la luz

    for (x0, y0, z0), E, alpha, beta, unit_vec, unit_vector_3D in Electron_entrances:
        E_J = E * 1.602e-13
        gamma = 1 + E_J / (m * c**2)
        v = c * np.sqrt(1 - (1 / gamma)**2)
        r = (gamma * m * v) / (q * B)

        if r >= yf:
            theta = np.arccos(np.cos(beta) - (yf - y0) / r)
            xf = x0 + r * (np.sin(theta) - np.sin(beta))

            a_z = q * E_z / (gamma * m)
            arc_length = r * theta
            t_arc = arc_length / v

            # Componentes de velocidad
            v_perp = v * np.sqrt(unit_vector_3D[0]**2 + unit_vector_3D[1]**2)
            v_par = v * unit_vector_3D[2]

            vx = -v_perp * np.sin(theta)
            vy =  v_perp * np.cos(theta)

            if E_z == 0:
                vz = v_par
                zf = z0 + vz * t_arc
            else:
                vz = v_par + a_z * t_arc
                zf = z0 + v_par * t_arc + 0.5 * a_z * t_arc**2

            if np.abs(zf) < 0.0025:
                exit_vec = np.array([vx, vy, vz])
                exit_unit_vector_3D = exit_vec / np.linalg.norm(exit_vec)
                dir_xf = np.cos(theta)
                dir_yf = np.sin(theta)

                Electron_Exits.append(((x0, y0, z0), (xf, yf, zf), (dir_xf, dir_yf), E, r, theta, alpha, beta, a_z, t_arc, vz, unit_vector_3D, exit_unit_vector_3D))

    return Electron_Exits

def Intersections(Electron_Exits, yff):
    intersections = []
    m = 9.109e-31
    c = 3e8
    for (x0, y0, z0), (xf, yf, zf), (dir_xf, dir_yf), E, r, theta, alpha, beta, a_z, t_arc, v_z, unit_vector_3D, exit_unit_vector_3D in Electron_Exits:
        E_J = E * 1.60218e-13
        gamma = 1 + E_J / (m * c**2)
        v = c * np.sqrt(1 - (1 / gamma)**2)
        delta_y_line = yff - yf
        t_line = delta_y_line / (v * np.sin(theta))
        xff = xf + v * np.cos(theta) * t_line
        vz_exit = v * exit_unit_vector_3D[2]
        zff = zf + vz_exit * t_line
        intersections.append(((x0, y0, z0), (xf, yf, zf), (xff, yff, zff), E, r, theta, alpha, beta, unit_vector_3D, exit_unit_vector_3D))
    return intersections

def compute_trajectory(start_point, r, alpha, theta, beta, yff, E, unit_vector_3D, exit_unit_vector_3D, num_points=100, E_z=0):
    x0, y0, z0 = start_point
    q = 1.602e-19
    m = 9.109e-31
    c = 3e8
    E_J = E * 1.602e-13
    gamma = 1 + E_J / (m * c**2)
    v = c * np.sqrt(1 - (1 / gamma)**2)
    a_z = q * E_z / (gamma * m)

    angles = np.linspace(beta, beta + theta, num_points)
    x_arc = x0 + r * (np.sin(angles) - np.sin(beta))
    y_arc = y0 - r * (np.cos(angles) - np.cos(beta))

    arc_length = r * theta
    t_arc = arc_length / v
    t_vals = np.linspace(0, t_arc, num_points)
    z_arc = z0 + v * unit_vector_3D[2] * t_vals + a_z * t_vals**2 / 2

    # *** FILTRAR TRAJECTORIAS CON z_arc FUERA DE RANGO ***
    if np.any(np.abs(z_arc) >= 0.0025):
        return np.array([]), np.array([]), np.array([])

    dx = x_arc[-1] - x_arc[-2]
    dy = y_arc[-1] - y_arc[-2]
    dz = z_arc[-1] - z_arc[-2]
    tangent_vector = np.array([dx, dy, dz])
    tangent_unit = tangent_vector / np.linalg.norm(tangent_vector)

    xf, yf = x_arc[-1], y_arc[-1]
    delta_y_line = yff - yf
    t_line = delta_y_line / (v * np.sin(theta))
    xff = xf + v * np.cos(theta) * t_line
    zff = z_arc[-1] + v * exit_unit_vector_3D[2] * t_line

    if E_z == 0:
        x_line = np.array([xf, xff])
        y_line = np.array([yf, yff])
        dt = t_arc / num_points
        v_z = (z_arc[-1] - z_arc[-2]) / dt
        z_line = np.array([z_arc[-1], zff])
    else:
        delta_y_line = yff - y_arc[-1]
        t_line = delta_y_line / (v * exit_unit_vector_3D[1])

        x_line = np.array([x_arc[-1], x_arc[-1] + v * exit_unit_vector_3D[0] * t_line])
        y_line = np.array([y_arc[-1], yff])
        z_line = np.array([z_arc[-1], z_arc[-1] + v * exit_unit_vector_3D[2] * t_line])
        arc_mask = x_arc <= 0.18
        x_arc, y_arc, z_arc = x_arc[arc_mask], y_arc[arc_mask], z_arc[arc_mask]

    x_total = np.concatenate((x_arc, x_line))
    y_total = np.concatenate((y_arc, y_line))
    z_total = np.concatenate((z_arc, z_line))

    return x_total, y_total, z_total

electron_quantity = 10000000
B_field_strength = 0.07
x_entrada = 0.025
yf = 0.02
yff = 0.05
E_z = 0
x_i, y_i, z_i, Energies_D = Inicialization(electron_quantity)
entrances = ElectronEntrancePoint_Camera(x_i, y_i, z_i, Energies_D, x_entrada)
exits = ElectronExitPoint(entrances, B_field_strength, yf, E_z)
intersections = Intersections(exits, yff)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
all_energies = [e[3] for e in intersections]
norm = colors.Normalize(vmin=min(all_energies), vmax=max(all_energies))
cmap = cm.plasma

for inter in intersections:
    (x0, y0, z0), (xf, yf, zf), (xff, yff, zff), E, r, theta, alpha, beta, unit_vector_3D, exit_unit_vector_3D = inter
    x_vals, y_vals, z_vals = compute_trajectory((x0, y0, z0), r, alpha, theta, beta, yff, E, unit_vector_3D, exit_unit_vector_3D, E_z=E_z)
    color = cmap(norm(E))
    ax.plot(x_vals, y_vals, z_vals, color=color, alpha=0.5)

ax.set_xlabel('x [m]', labelpad=15)
ax.set_ylabel('y [m]', labelpad=15)
ax.set_zlabel('z [m]', labelpad=20)
ax.set_title('Ray Tracing de electrones en 3D')

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Energía [MeV]')

# Crear planos sólidos en z = 0.0025 y z = -0.0025
#x_plane = np.linspace(0.025, 0.125, 2)
#y_plane = np.linspace(0.0, 0.02, 2)
#x_plane, y_plane = np.meshgrid(x_plane, y_plane)

# Plano superior en z = 0.0025
#z_plane1 = np.full_like(x_plane, 0.0025)
#ax.plot_surface(x_plane, y_plane, z_plane1, color='gray')

# Plano inferior en z = -0.0025
#z_plane2 = np.full_like(x_plane, -0.0025)
#ax.plot_surface(x_plane, y_plane, z_plane2, color='gray')

ax.zaxis.set_tick_params(pad=12)

plt.tight_layout()

# Gráfico 2D del plano y = yff con puntos (xff, zff)
fig2, ax2 = plt.subplots(figsize=(8, 6))

xffs = [inter[2][0] for inter in intersections]
zffs = [inter[2][2] for inter in intersections]
energies = [inter[3] for inter in intersections]

sc = ax2.scatter(xffs, zffs, c=energies, cmap=cmap, norm=norm, s=5, alpha=0.7)
ax2.set_xlabel('x [m]')
ax2.set_ylabel('z [m]')
ax2.set_title('Distribución espacial de electrones sobre la Image Plate')
cbar2 = plt.colorbar(sc, ax=ax2)
cbar2.set_label('Energía [MeV]')

plt.tight_layout()

# Crear tercer gráfico: xff vs zff para y = 0.05
fig2, ax2 = plt.subplots(figsize=(10, 6))

# Extraer datos de intersección
xff_values = []
zff_values = []
energies = []

for inter in intersections:
    (_, _, _), (_, _, _), (xff, _, zff), E, *_ = inter
    xff_values.append(xff)
    zff_values.append(zff)
    energies.append(E)

xff_values = np.array(xff_values)
zff_values = np.array(zff_values)
energies = np.array(energies)

# Definir energías de interés
energies_of_interest = np.arange(0.1, 1.55, 0.1)  # Desde 0.5 a 1.5 MeV
delta_E = 0.05

# Colormap
cmap = plt.get_cmap('plasma')
norm = colors.Normalize(vmin=min(energies_of_interest)-0.05, vmax=max(energies_of_interest)+0.05)

# Dibujar puntos de intersección
sc = ax2.scatter(xff_values, zff_values, c=energies, cmap=cmap, norm=norm, s=5, alpha=0.6)

# Dibujar franjas para energías de interés
for E_target in energies_of_interest:
    mask = (energies >= E_target - delta_E) & (energies <= E_target + delta_E)
    x_vals = xff_values[mask]
    
    if len(x_vals) > 0:
        x_mean = np.mean(x_vals)
        x_std = np.std(x_vals)
        
        ax2.axvspan(x_mean - x_std, x_mean + x_std, color=cmap(norm(E_target)), alpha=0.2,
                    label=f'{E_target:.1f} MeV')

# Ajustes del gráfico
#ax2.set_xlabel('x [m]')
#ax2.set_ylabel('z [m]')
#ax2.set_title('Intersección en y = 0.05 m: x vs z')
#ax2.legend(loc='upper left', title='Energía (MeV)', bbox_to_anchor=(1.2, 1), borderaxespad=0.)
#plt.colorbar(sc, ax=ax2, label='Energía (MeV)', orientation='vertical')

#plt.tight_layout()
#plt.show()

# Crear tercer gráfico: xff vs zff para y = 0.05
fig2, ax2 = plt.subplots(figsize=(10, 6))

# Extraer datos de intersección
xff_values = []
zff_values = []
energies = []

for inter in intersections:
    (_, _, _), (_, _, _), (xff, _, zff), E, *_ = inter
    xff_values.append(xff)
    zff_values.append(zff)
    energies.append(E)

xff_values = np.array(xff_values)
zff_values = np.array(zff_values)
energies = np.array(energies)

# Definir energías de interés
energies_of_interest = np.arange(0.1, 1.7, 0.1)  # Desde 0.1 a 1.5 MeV
delta_E = 0.05

# Colormap
cmap = plt.get_cmap('plasma')
norm = colors.Normalize(vmin=min(energies_of_interest)-delta_E, vmax=max(energies_of_interest)+delta_E)

# Dibujar puntos de intersección
sc = ax2.scatter(xff_values, zff_values, c=energies, cmap=cmap, norm=norm, s=5, alpha=0.6)

# Dibujar franjas para energías de interés
# Dibujar franjas para energías de interés
for E_target in energies_of_interest:
    mask = (energies >= E_target - delta_E) & (energies <= E_target + delta_E)
    x_vals = xff_values[mask]

    if len(x_vals) > 0:
        x_min = np.min(x_vals)
        x_max = np.max(x_vals)
        x_mean = np.mean(x_vals)
        franja_ancho = round((x_max - x_min)*100, 1) 

        # Franja de mínimos y máximos
        ax2.axvspan(x_min, x_max, color=cmap(norm(E_target)), alpha=0.2,
                    label=f'{E_target:.1f} MeV')

        # Línea punteada en el promedio
        ax2.axvline(x_mean, color=cmap(norm(E_target)), linestyle='--', linewidth=1)

        y_max = ax2.get_ylim()[1]
        ax2.text(x_mean, y_max * 1.01, f'{E_target:.1f} MeV',
                color=cmap(norm(E_target)), ha='center', va='bottom', rotation=90)

        # Texto con ancho de la franja
        #y_text = np.min(zff_values) - 0.005  # Un poco debajo de los datos
        #ax2.text(x_mean, y_text, f'{franja_ancho} cm', ha='top', va='top', fontsize=8)


# Ajustes del gráfico
ax2.set_xlabel('x [m]')
ax2.set_ylabel('z [m]')
ax2.set_title('Distribución espacial de electrones sobre la Image Plate', pad=65)

#ax2.legend(loc='upper left', title='Energía (MeV)', bbox_to_anchor=(1.2, 1), borderaxespad=0.)
plt.colorbar(sc, ax=ax2, label='Energía [MeV]', orientation='vertical')

plt.tight_layout()

# Obtener energías de los electrones que llegan a y = 0.05
energies_at_yff = [E for _, _, _, E, *_ in intersections]

# Crear histograma: número de electrones vs energía

# Centros deseados: 0.1, 0.2, ..., 1.6
bin_centers = np.arange(0.1, 1.61, 0.1)
bin_edges = np.arange(0.05, 1.66, 0.1)  # Bordes correspondientes
# Extraer las posiciones xff de cada electrón que llega a y = 0.05
xff_values = np.array([intersection[2][0] for intersection in intersections])

# Crear histograma
bins_edges = np.linspace(0.04, 0.165, 45)  # puedes ajustar el número de bins

plt.figure(figsize=(8, 5))
plt.hist(energies_at_yff, bins=bin_edges, color='skyblue', edgecolor='black')
plt.xlabel('Energía [MeV]')
plt.ylabel('Número de electrones')
plt.title('Distribución de energía de electrones sobre la Image Plate')
plt.grid(True)
plt.tight_layout()

# Graficar
#plt.figure(figsize=(10,6))
#plt.hist(xff_values, bins=bins_edges, color='blue', edgecolor='black')
#plt.xlabel('Posición en x [m] en y = 0.05 m')
#plt.ylabel('Número de electrones')
#plt.title('Distribución de electrones vs posición en x en el plano y = 0.05 m')
#plt.grid(True)
#plt.tight_layout()

centers = []
heights = []
widths = []

x_vals_dict = {}
for E_target in energies_of_interest:
    mask = (energies >= E_target - delta_E) & (energies <= E_target + delta_E)
    x_vals = xff_values[mask]
    if len(x_vals) > 0:
        x_vals_dict[E_target] = np.sort(x_vals)

sorted_energies = sorted(x_vals_dict.keys())

for i, E in enumerate(sorted_energies):
    x_vals = x_vals_dict[E]
    x_min = x_vals[0]
    x_max = x_vals[-1]
    x_mean = x_vals.mean()

    # Determinar bordes sin solapamiento
    if i > 0:
        prev_x_max = x_vals_dict[sorted_energies[i - 1]][-1]
        left_edge = max(x_min, prev_x_max)
    else:
        left_edge = x_min

    if i < len(sorted_energies) - 1:
        next_x_min = x_vals_dict[sorted_energies[i + 1]][0]
        right_edge = min(x_max, next_x_min)
    else:
        right_edge = x_max

    width = right_edge - left_edge

    # Ahora filtramos los x_vals dentro del rango visual
    valid_vals = x_vals[(x_vals >= left_edge) & (x_vals <= right_edge)]

    if width > 0 and len(valid_vals) > 0:
        centers.append(np.mean(valid_vals))  # x_mean dentro del rango visible
        widths.append(width)
        heights.append(len(valid_vals))

# Graficar
#plt.figure(figsize=(10,6))
#plt.bar(centers, heights, width=widths, align='center', edgecolor='black', color='steelblue')
#plt.xlabel('Posición en x [m]')
#plt.ylabel('Número de electrones (sin solapamiento)')
#plt.title('Distribución corregida de electrones vs posición xff')
#plt.grid(True)
#plt.tight_layout()


# Tu array de posiciones finales en x
xff_values = np.array([intersection[2][0] for intersection in intersections])
energies = np.array([intersection[3] for intersection in intersections])  # Energía de cada electrón

# Parámetros

# Crear figura y eje
fig, ax = plt.subplots(figsize=(10, 6))

# Histograma general
ax.hist(xff_values, bins=bins_edges, color='blue', edgecolor='black')

# Superponer franjas y líneas por energía
for E_target in energies_of_interest:
    mask = (energies >= E_target - delta_E) & (energies <= E_target + delta_E)
    x_vals = xff_values[mask]

    if len(x_vals) > 0:
        x_min = np.min(x_vals)
        x_max = np.max(x_vals)
        x_mean = np.mean(x_vals)

        # Franja entre x_min y x_max
        ax.axvspan(x_min, x_max, color=cmap(norm(E_target)), alpha=0.2,
                   label=f'{E_target:.1f} MeV')

        # Línea punteada en el promedio
        ax.axvline(x_mean, color=cmap(norm(E_target)), linestyle='--', linewidth=1)

        # Texto con energía sobre la línea del promedio
        # Para ubicarlo, calculamos la altura máxima local
        y_max = ax.get_ylim()[1]
        ax.text(x_mean, y_max * 1.01, f'{E_target:.1f} MeV',
                color=cmap(norm(E_target)), ha='center', va='bottom', rotation=90)

# Ajustes finales
ax.set_xlabel('Posición en x sobre la Image Plate [m]')
ax.set_ylabel('Número de electrones')
ax.set_title('Distribución espacial de electrones en x sobre la Image Plate', pad=65)
ax.grid(True)
#ax.legend(title="Energía central", fontsize='small', title_fontsize='medium', loc='upper right', bbox_to_anchor=(1.25, 1))
plt.tight_layout()
plt.show()