import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
import qubits

def slerp_with_matrix(v0, v1, t, matrix):
    """
    Spherical Linear Interpolation that handles 180-degree rotations 
    by extracting the exact rotation axis from the transformation matrix.
    """
    v0_norm = v0 / (np.linalg.norm(v0) + 1e-9)
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-9)
    
    dot = np.clip(np.dot(v0_norm, v1_norm), -1.0, 1.0)
    
    if dot > 0.9995:
        v_curr = (1.0 - t) * v0_norm + t * v1_norm
        return v_curr / np.linalg.norm(v_curr)
        
    elif dot > -0.9995:
        theta_0 = np.arccos(dot)
        theta = theta_0 * t
        ortho = v1_norm - v0_norm * dot
        ortho = ortho / np.linalg.norm(ortho)
        return v0_norm * np.cos(theta) + ortho * np.sin(theta)
        
    else:
        M_plus_I = matrix + np.eye(3)
        col_norms = np.linalg.norm(M_plus_I, axis=0)
        max_col_idx = np.argmax(col_norms)
        
        if col_norms[max_col_idx] > 1e-5:
            axis = M_plus_I[:, max_col_idx]
            axis = axis / np.linalg.norm(axis)
        else:
            temp = np.array([1.0, 0.0, 0.0])
            if np.abs(np.dot(v0_norm, temp)) > 0.99:
                temp = np.array([0.0, 1.0, 0.0])
            axis = temp - np.dot(v0_norm, temp) * v0_norm
            axis = axis / np.linalg.norm(axis)
            
        cross_vec = np.cross(axis, v0_norm)
        return v0_norm * np.cos(t * np.pi) + cross_vec * np.sin(t * np.pi)


def animate_transformation(v_initials, matrices, colors=None):
    """
    Animates the sequential transformation of state vectors, each in its own unit sphere.
    
    Args:
        v_initials (list, np.ndarray, or list of lists): The initial 3D vector(s).
        matrices (list of lists or list of np.ndarray): A list of 3x3 transformation matrices.
        colors (list of str): Optional custom colors for each vector.
    """
    v_initials = np.array(v_initials, dtype=float)
    if v_initials.ndim == 1:
        v_initials = np.array([v_initials])
        
    matrices = [np.array(m, dtype=float) for m in matrices]
    num_vectors = len(v_initials)
    
    if colors is None:
        default_colors = ['b', 'g', 'm', 'c', 'orange', 'purple', 'brown']
        colors = [default_colors[i % len(default_colors)] for i in range(num_vectors)]

    radius = 1.0
    u = np.linspace(0, 2 * np.pi, 25)
    v = np.linspace(0, np.pi, 25)
    x_sphere = radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = radius * np.outer(np.ones(np.size(u)), np.cos(v))

    # Dynamically scale width based on the number of vectors
    fig_width = max(5 * num_vectors, 6)
    fig = plt.figure(figsize=(fig_width, 5.5), num='Bloch Sphere Animation')
    plt.subplots_adjust(bottom=0.25, wspace=0.1) 
    
    # Generate subplots dynamically
    axes = []
    for i in range(num_vectors):
        ax = fig.add_subplot(1, num_vectors, i + 1, projection='3d')
        axes.append(ax)

    state = {
        'current_t': 0.0,
        'is_playing': False,
        'ani': None,
        'matrix_idx': 0,
        'v_starts': [v.copy() for v in v_initials],
        'v_ends': [matrices[0] @ v for v in v_initials],
        'v_rendereds': [v.copy() for v in v_initials]
    }

    # UI Layout centered at the bottom
    ax_button = plt.axes([0.35, 0.12, 0.1, 0.06])
    btn = Button(ax_button, 'Play')
    
    ax_measure = plt.axes([0.55, 0.12, 0.1, 0.06])
    btn_measure = Button(ax_measure, 'Measure')
    
    ax_slider = plt.axes([0.25, 0.04, 0.5, 0.03])
    speed_slider = Slider(ax_slider, 'Speed', 0.1, 3.0, valinit=1.0, valfmt="%0.1fx")

    def update(frame):
        lim = 1.2
        current_matrix = matrices[state['matrix_idx']]
        
        # Update global title text
        fig.suptitle(f'Step {state["matrix_idx"] + 1}/{len(matrices)} ({state["current_t"]*100:.0f}%)', fontsize=14)

        # Draw each sphere and its corresponding vector
        for i, ax in enumerate(axes):
            ax.cla()  
            ax.axis('off')
            
            # Setup basis labels and lines
            ax.plot([-lim, lim], [0, 0], [0, 0], color='black', linewidth=1, alpha=0.8)
            ax.plot([0, 0], [-lim, lim], [0, 0], color='black', linewidth=1, alpha=0.8)
            ax.plot([0, 0], [0, 0], [-lim, lim], color='black', linewidth=1, alpha=0.8)
            ax.text( lim + 0.1, 0, 0, '|+>', color='black', fontsize=10)
            ax.text(-lim - 0.1, 0, 0, '|->', color='black', fontsize=10)
            ax.text(0,  lim + 0.1, 0, '|i>', color='black', fontsize=10)
            ax.text(0, -lim - 0.1, 0, '|-i>', color='black', fontsize=10)
            ax.text(0, 0, lim + 0.1, '|0>', color='black', fontsize=10)
            ax.text(0, 0,-lim - 0.1, '|1>', color='black', fontsize=10)
            
            ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.15)
            
            # SLERP calculations
            v_current = slerp_with_matrix(state['v_starts'][i], state['v_ends'][i], state['current_t'], current_matrix)
            state['v_rendereds'][i] = v_current.copy()
            
            ax.quiver(0, 0, 0, v_current[0], v_current[1], v_current[2], color=colors[i], arrow_length_ratio=0.1)
            ax.quiver(0, 0, 0, v_initials[i][0], v_initials[i][1], v_initials[i][2], color=colors[i], alpha=0.2, arrow_length_ratio=0.1)
            
            ax.set_xlim([-lim, lim])
            ax.set_ylim([-lim, lim])
            ax.set_zlim([-lim, lim])
            ax.set_box_aspect([1, 1, 1])
            ax.set_title(f'X Gate Example')

        # State management (runs once per frame, not per vector)
        if state['is_playing']:
            if state['current_t'] >= 1.0:
                state['matrix_idx'] += 1
                
                if state['matrix_idx'] < len(matrices):
                    state['current_t'] = 0.0
                    state['v_starts'] = [v.copy() for v in state['v_ends']]
                    state['v_ends'] = [matrices[state['matrix_idx']] @ v for v in state['v_starts']]
                else:
                    if state['ani']:
                        state['ani'].pause()
                    state['is_playing'] = False
                    btn.label.set_text('Play')
                    fig.canvas.draw_idle()
                    
                    state['current_t'] = 0.0 
                    state['matrix_idx'] = 0
                    state['v_starts'] = [v.copy() for v in v_initials]
                    state['v_ends'] = [matrices[0] @ v for v in state['v_starts']]
            else:
                # Doubled the frames by halving the step increment (from 1/60 to 1/120)
                step = (1.0 / 120.0) * speed_slider.val
                state['current_t'] = min(state['current_t'] + step, 1.0)

    def toggle(event):
        if state['is_playing']:
            if state['ani']:
                state['ani'].pause()
            btn.label.set_text('Play')
        else:
            if state['ani']:
                state['ani'].resume()
            btn.label.set_text('Pause')
        state['is_playing'] = not state['is_playing']
        fig.canvas.draw_idle() 

    def measure(event):
        current_matrix = matrices[state['matrix_idx']]
        
        if state['is_playing']:
            if state['ani']:
                state['ani'].pause()
            state['is_playing'] = False
            btn.label.set_text('Play')
            
        for i in range(num_vectors):
            v_current = state['v_rendereds'][i]
            v_collapsed = qubits.measure_qubit(v_current)
            
            state['v_starts'][i] = np.array(v_collapsed, dtype=float)
            state['v_ends'][i] = current_matrix @ state['v_starts'][i]
        
        state['current_t'] = 0.0
        update(0)
        fig.canvas.draw_idle()

    btn.on_clicked(toggle)
    btn_measure.on_clicked(measure)

    # Increased frames argument to 200 to accommodate the smaller step size
    state['ani'] = FuncAnimation(fig, update, frames=200, interval=16, repeat=True)

    def stop_initial(event):
        if not state['is_playing'] and state['ani']:
            state['ani'].pause()
        fig.canvas.mpl_disconnect(cid)
        
    cid = fig.canvas.mpl_connect('draw_event', stop_initial)

    plt.show()
    
    return fig, state['ani'], btn, btn_measure, speed_slider