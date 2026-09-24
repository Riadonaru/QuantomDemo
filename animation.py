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
    # Normalize to ensure we stay perfectly on the unit sphere
    v0_norm = v0 / (np.linalg.norm(v0) + 1e-9)
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-9)
    
    dot = np.clip(np.dot(v0_norm, v1_norm), -1.0, 1.0)
    
    if dot > 0.9995:
        # Linear interpolation for very close vectors
        v_curr = (1.0 - t) * v0_norm + t * v1_norm
        return v_curr / np.linalg.norm(v_curr)
        
    elif dot > -0.9995:
        # Standard SLERP for normal angles
        theta_0 = np.arccos(dot)
        theta = theta_0 * t
        ortho = v1_norm - v0_norm * dot
        ortho = ortho / np.linalg.norm(ortho)
        return v0_norm * np.cos(theta) + ortho * np.sin(theta)
        
    else:
        # Anti-parallel vectors (180 degree rotation)
        # We extract the axis of rotation directly from the matrix (M + I)
        M_plus_I = matrix + np.eye(3)
        col_norms = np.linalg.norm(M_plus_I, axis=0)
        max_col_idx = np.argmax(col_norms)
        
        if col_norms[max_col_idx] > 1e-5:
            axis = M_plus_I[:, max_col_idx]
            axis = axis / np.linalg.norm(axis)
        else:
            # Fallback if matrix is -I (all axes are valid)
            temp = np.array([1.0, 0.0, 0.0])
            if np.abs(np.dot(v0_norm, temp)) > 0.99:
                temp = np.array([0.0, 1.0, 0.0])
            axis = temp - np.dot(v0_norm, temp) * v0_norm
            axis = axis / np.linalg.norm(axis)
            
        # Rotate v0 around the extracted axis by (t * pi)
        cross_vec = np.cross(axis, v0_norm)
        return v0_norm * np.cos(t * np.pi) + cross_vec * np.sin(t * np.pi)


def animate_transformation(v_initial, matrices):
    """
    Animates the sequential transformation of a state vector on a unit sphere.
    
    Args:
        v_initial (list or np.ndarray): The initial 3D vector.
        matrices (list of lists or list of np.ndarray): A list of 3x3 transformation matrices.
    """
    v_initial = np.array(v_initial, dtype=float)
    matrices = [np.array(m, dtype=float) for m in matrices]

    radius = 1.0
    u = np.linspace(0, 2 * np.pi, 25)
    v = np.linspace(0, np.pi, 25)
    x_sphere = radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = radius * np.outer(np.ones(np.size(u)), np.cos(v))

    fig = plt.figure(figsize=(7, 7))
    plt.subplots_adjust(bottom=0.20) 
    ax = fig.add_subplot(111, projection='3d')

    state = {
        'current_t': 0.0,
        'is_playing': False,
        'ani': None,
        'matrix_idx': 0,
        'v_start': v_initial.copy(),
        'v_end': matrices[0] @ v_initial,
        'v_rendered': v_initial.copy()
    }

    # Adjusted Play Button Layout
    ax_button = plt.axes([0.3, 0.10, 0.15, 0.05])
    btn = Button(ax_button, 'Play')
    
    # New Measure Button
    ax_measure = plt.axes([0.55, 0.10, 0.15, 0.05])
    btn_measure = Button(ax_measure, 'Measure')
    
    ax_slider = plt.axes([0.2, 0.04, 0.6, 0.03])
    speed_slider = Slider(ax_slider, 'Speed', 0.1, 3.0, valinit=1.0, valfmt="%0.1fx")

    def update(frame):
        ax.cla()  
        ax.axis('off')
        
        lim = 1.2
        
        ax.plot([-lim, lim], [0, 0], [0, 0], color='black', linewidth=1, alpha=0.8)
        ax.plot([0, 0], [-lim, lim], [0, 0], color='black', linewidth=1, alpha=0.8)
        ax.plot([0, 0], [0, 0], [-lim, lim], color='black', linewidth=1, alpha=0.8)
        ax.text( lim + 0.1, 0, 0, '|+>', color='black', fontsize=10)
        ax.text(-lim - 0.1, 0, 0, '|->', color='black', fontsize=10)
        ax.text(0, lim + 0.1, 0, 'Y', color='black', fontsize=10)
        ax.text(0, 0, lim + 0.1, '|0>', color='black', fontsize=10)
        ax.text(0, 0,-lim - 0.1, '|1>', color='black', fontsize=10)
        
        ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.15)
        
        # SLERP handles 180 degree flips properly without going through origin
        current_matrix = matrices[state['matrix_idx']]
        v_current = slerp_with_matrix(state['v_start'], state['v_end'], state['current_t'], current_matrix)
        state['v_rendered'] = v_current.copy()
        
        ax.quiver(0, 0, 0, v_current[0], v_current[1], v_current[2], color='b', arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, v_initial[0], v_initial[1], v_initial[2], color='r', alpha=0.2, arrow_length_ratio=0.1)
        
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        ax.set_zlim([-lim, lim])
        ax.set_box_aspect([1, 1, 1])
        ax.text2D(0.5, 0.95, f'Step {state["matrix_idx"] + 1}/{len(matrices)} ({state["current_t"]*100:.0f}%)', 
                  transform=ax.transAxes, horizontalalignment='center', fontsize=12)

        if state['is_playing']:
            if state['current_t'] >= 1.0:
                state['matrix_idx'] += 1
                
                if state['matrix_idx'] < len(matrices):
                    state['current_t'] = 0.0
                    state['v_start'] = state['v_end'].copy()
                    state['v_end'] = matrices[state['matrix_idx']] @ state['v_start']
                else:
                    if state['ani']:
                        state['ani'].pause()
                    state['is_playing'] = False
                    btn.label.set_text('Play')
                    fig.canvas.draw_idle()
                    
                    state['current_t'] = 0.0 
                    state['matrix_idx'] = 0
                    state['v_start'] = v_initial.copy()
                    state['v_end'] = matrices[0] @ state['v_start']
            else:
                step = (1.0 / 60.0) * speed_slider.val
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
        # Calculate current vector based on animation state
        current_matrix = matrices[state['matrix_idx']]
        v_current = state['v_rendered']  # Use the last rendered vector for measurement
        v_collapsed = qubits.measure_qubit(v_current)
        
        # Stop playback
        if state['is_playing']:
            if state['ani']:
                state['ani'].pause()
            state['is_playing'] = False
            btn.label.set_text('Play')
            
        # Snap vector to result
        state['v_start'] = np.array(v_collapsed, dtype=float)
        
        # FIX: Set the destination to be the current transformation applied to the collapsed state
        state['v_end'] = current_matrix @ state['v_start'] 
        
        state['current_t'] = 0.0
        
        # Force a frame update immediately to show the collapsed state
        update(0)
        fig.canvas.draw_idle()

    btn.on_clicked(toggle)
    btn_measure.on_clicked(measure)

    state['ani'] = FuncAnimation(fig, update, frames=100, interval=16, repeat=True)

    def stop_initial(event):
        if not state['is_playing'] and state['ani']:
            state['ani'].pause()
        fig.canvas.mpl_disconnect(cid)
        
    cid = fig.canvas.mpl_connect('draw_event', stop_initial)

    plt.show()
    
    # Returning btn_measure ensures the button is not garbage collected
    return fig, state['ani'], btn, btn_measure, speed_slider