import matplotlib.pyplot as plt
import pyvista as pv
import numpy as np


def viz(data,delay):
    # Inspect shape
    print("Shape:", data.shape)  # Should be (Depth, Height, Width)

    for i in range(data.shape[0]):
        plt.imshow(data[i, :, :], cmap='gray')
        plt.title(f"Slice {i}")
        plt.pause(delay)
        plt.clf()
        
    # Show the last slice and wait until window closed
    plt.imshow(data[-1, :, :], cmap='gray')
    plt.title(f"Slice {data.shape[0]-1}")
    plt.show()  # This blocks until you close the window
    
def vizMid(data):
    # Show a middle slice (you can change axis or index)
    plt.imshow(data[data.shape[0] // 2, :, :], cmap='gray')
    plt.title("Middle axial slice")
    plt.axis('off')
    plt.show()
    
def viz3D(data):
    grid = pv.ImageData()

    # Set the grid dimensions: shape + 1 because we want to inject our values on
    #   the CELL data
    grid.dimensions = np.array(data.shape) + 1

    grid.spacing = (1, 1, 1)  # These are the cell sizes along each axis

    # Add the data values to the cell data
    grid.cell_data['values'] = data.flatten(order='F')  # Flatten the array

    # Plot
    plotter = pv.Plotter()
    plotter.add_volume(grid, cmap="gray",opacity="linear")
    plotter.show()
    
import pyvista as pv
import numpy as np

def viz3D_with_slider(data4d):
    """
    Visualize 4D volume data (T, X, Y, Z) using PyVista with a slider.

    Parameters:
    - data4d: 4D NumPy array of shape (T, X, Y, Z)
    """


    T = len(data4d)

    data0 = data4d[0]

    # Setup volume grid
    grid = pv.ImageData()
    grid.dimensions = np.array(data0.shape) + 1
    grid.spacing = (1, 1, 1)
    grid.cell_data['values'] = data0.flatten(order='F')

    # Create plotter
    plotter = pv.Plotter()
    actor = plotter.add_volume(grid, cmap='gray', opacity='linear')
    plotter.add_title(f"Time Index: 0")
    plotter.show_axes()
    plotter.show_bounds()
    # Callback to update volume when slider moves
    def update_volume(time_idx):
        nonlocal actor
        t = int(time_idx)
        volume_t = data4d[t]
        grid.cell_data['values'] = volume_t.flatten(order='F')
        plotter.remove_actor(actor)
        actor=plotter.add_volume(grid, cmap='gray', opacity='linear')
        plotter.add_title(f"Time Index: {t}")
        plotter.render()

    # Add slider widget
    plotter.add_slider_widget(
        callback=update_volume,
        rng=[0, T - 1],
        value=0,
        title='Time',
        interaction_event='always',
    )

    plotter.show()




    