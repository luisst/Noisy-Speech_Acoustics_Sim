import matplotlib.pyplot as plt

def plot_squares(squares, x_min=None, x_max=None, y_min=None, y_max=None, colors=None):
    """
    Plots a list of squares based on the 2 tuples of coordinates provided for
    the lower and upper corners of each square using matplotlib, with a grid
    of 10 cm (0.1 meters).
    """
    fig, ax = plt.subplots()


    for i, (lower_corner, upper_corner) in enumerate(squares):
        x1, y1 = lower_corner
        x2, y2 = upper_corner

        # Calculate the width and height of the square
        width = abs(x2 - x1)
        height = abs(y2 - y1)

        # Add the square
        if colors:
            square = plt.Rectangle(lower_corner, width, height, fill=False, edgecolor=colors[i], linewidth=2)
        else:
            square = plt.Rectangle(lower_corner, width, height, fill=False, edgecolor='blue', linewidth=2)
        ax.add_patch(square)

    # Set the axis limits with extra space around the squares
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


    # Set the major and minor ticks for the x and y axes
    ax.set_xticks(range(x_max + 1), minor=True)
    ax.set_yticks(range(y_max + 1), minor=True)
    ax.set_xticks([i + 0.5 for i in range(x_max + 1)], minor=False)
    ax.set_yticks([i + 0.5 for i in range(y_max + 1)], minor=False)

    # Set the grid
    ax.grid(which='major', linestyle=':', linewidth=0.5)
    ax.grid(which='minor', linestyle=':', linewidth=0.5)

    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')

    return fig, ax


def plot_marks(fig, ax, x_marks):
    """
    Adds X marks at the specified coordinates (x, y) to the given
    figure and axis objects.
    """
    for x, y in x_marks:
        ax.plot(x, y, 'x', color='blue', markersize=10)


def plot_single_configuration(single_cfg, filename=None, legend=None):
    colors = ['black', 'blue', 'green', 'green'] # Specifies the color of each square

    fig, ax = plot_squares(single_cfg['regions'], x_min=0, 
                        x_max=int(single_cfg['room'][0]), y_min=0, 
                        y_max=int(single_cfg['room'][1]), colors=colors) 

    # Plot microphone
    ax.plot(single_cfg['mic'][0], single_cfg['mic'][1], 'o', color='red', markersize=10)

    # Plot main speaker
    ax.plot(single_cfg['main'][0], single_cfg['main'][1], 'D', color='green', markersize=10)

    # Plot noise
    ax.plot(single_cfg['noise'][0], single_cfg['noise'][1], '^', color='purple', markersize=10)

    plot_marks(fig, ax, single_cfg['others']) # Adds X marks to the figure and axis objects


    if filename:
        plt.savefig(filename, dpi=300)

    # plt.show()

if __name__ == "__main__":
    single_cfg = {'mic': [5,5],
                'main': [2, 4],
                'noise': [4,6],
                'others': [(2.5, 3), (4,2.5)],
                'regions': [((1,1),(5.5,6)),
                            ((3,3),(5,5))],
                'room': [10, 11]}


    plot_single_configuration(single_cfg, filename='holi.png')   
