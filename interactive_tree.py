# Import the pandas library, which is useful for working with structured data (like tables)
import pandas as pd
# Import the numpy library, which provides support for numerical operations and random number generation
import numpy as np
# Import the graph_objects module from Plotly, which is used to create interactive plots
import plotly.graph_objects as go
# Import the math library for mathematical functions like cosine, sine, and π
import math

# Define a function to generate an interactive visualization of a glucose prediction tree.
# The function takes in a DataFrame 'df', a file path 'save_path' for the output HTML, and a 'depth' to control tree levels.
def generate_interactive_glucose_tree(df, save_path="glucose_tree_interactive.html", depth=2):
    # This line above that contains the depth parameter being 2 or whatever value will always be overidden with the value decided in terminal
    # Create empty lists to store the x and y coordinates of nodes (points) in the tree.
    node_x = []
    node_y = []
    # Create an empty list to store text information for each node.
    node_text = []
    # Create an empty list to store color values for each node.
    node_color = []
    # Create empty lists to store the x and y coordinates of the lines (edges) connecting nodes.
    edge_x = []
    edge_y = []

    # Define a helper function that determines the color of a node based on the glucose value.
    def glucose_color(value):
        # If the glucose value is less than 80, return 'blue' to indicate a low value.
        if value < 80:
            return 'blue'
        # If the glucose value is greater than 130, return 'red' to indicate a high value.
        elif value > 130:
            return 'red'
        # For values in between, return 'green' to indicate a normal range.
        else:
            return 'green'

    # Define a recursive function to draw branches of the tree.
    # This function creates nodes and edges while reducing the depth until it reaches zero.
    def draw_branch(x, y, angle, depth, length, glucose_value, time_step):
        # Base case: if the depth is 0, stop drawing further branches.
        if depth == 0:
            return

        # Calculate the horizontal change (dx) using the cosine of the angle multiplied by the branch length.
        dx = length * math.cos(angle)
        # Calculate the vertical change (dy) using the sine of the angle multiplied by the branch length.
        dy = length * math.sin(angle)
        # Determine the new x-coordinate by adding dx to the current x.
        new_x = x + dx
        # Determine the new y-coordinate by adding dy to the current y.
        new_y = y + dy

        # Extend the edge lists with the current and new coordinates.
        edge_x.extend([x, new_x])
        edge_y.extend([y, new_y])
        # Add the new coordinates to the node lists.
        node_x.append(new_x)
        node_y.append(new_y)
        # Add text for the node showing the glucose value and the time step (formatted with a line break).
        node_text.append(f"Glucose: {int(glucose_value)} mg/dL<br>Time: T+{time_step}min")
        # Determine the node color based on the glucose value using the glucose_color function.
        node_color.append(glucose_color(glucose_value))

        # Set a seed for random number generation to ensure consistent results.
        # The seed is based on the current depth and the integer part of the glucose value.
        np.random.seed(depth * 10 + int(glucose_value))
        # Generate three random variations from a normal distribution (mean 0, standard deviation 10)
        # This simulates small changes in the glucose value.
        variations = np.random.normal(0, 10, 3)
        # Define three possible angles for new branches:
        # One branch deviates slightly to the right, one slightly to the left, and one continues straight.
        angles = [angle + 0.3, angle - 0.3, angle]

        # Loop over the three new branches.
        for i in range(3):
            # Update the glucose value for the new branch by adding the variation.
            new_val = glucose_value + variations[i]
            # Recursively draw the next branch:
            # - Start from the new node's position (new_x, new_y)
            # - Use the adjusted angle for this branch
            # - Reduce the depth by 1
            # - Reduce the branch length (multiplied by 0.75) for a smaller branch
            # - Update the glucose value and increase the time step by 5 minutes.
            draw_branch(new_x, new_y, angles[i], depth - 1, length * 0.75, new_val, time_step + 5)

    # Get the starting (root) glucose value from the DataFrame.
    # This uses the first row of the "glucose" column.
    root_glucose = df["glucose"].iloc[0]
    # Begin drawing the tree from the origin (0, 0) with an upward angle (pi/2 radians).
    # 'depth' is the tree's depth, 'length' is the initial branch length, and 'time_step' starts at 0.
    draw_branch(0, 0, math.pi / 2, depth=depth, length=1.5, glucose_value=root_glucose, time_step=0)

    # Create a new Plotly figure to hold the visualization.
    fig = go.Figure()

    # Add a trace (a series of connected data points) for the edges of the tree.
    # These are drawn as white lines and will not display any hover information.
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',  # Display as lines connecting the points
        line=dict(color='white', width=1),  # Set line color and width
        hoverinfo='none'  # Disable hover info for lines
    ))

    # Add another trace for the nodes (data points) of the tree.
    # These nodes will display both markers and text.
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',  # Show markers with text labels
        textposition='top center',  # Position the text above each marker
        marker=dict(color=node_color, size=8),  # Set the marker colors and size
        # Use only the first part of node text for visible labels (before the <br>)
        text=[text.split('<br>')[0] for text in node_text],
        hovertext=node_text,  # Full text (including time) will appear when hovered over
        hoverinfo='text'  # Enable hover information
    ))

    # Update the layout of the figure to enhance the visual style.
    fig.update_layout(
        title="🧠 Interactive Glucose Prediction Tree (ARIMA)",  # Set the title of the plot
        title_font_size=18,  # Set the font size for the title
        plot_bgcolor='black',  # Set the background color for the plot area to black
        paper_bgcolor='black',  # Set the background color for the entire page to black
        font=dict(color='white'),  # Set the default font color to white for visibility
        showlegend=False,  # Hide the legend since it is not needed
        # Hide the x-axis details (ticks, labels, etc.)
        xaxis=dict(visible=False),
        # Hide the y-axis details (ticks, labels, etc.)
        yaxis=dict(visible=False)
    )

    # Write the figure to an HTML file so it can be viewed in a web browser.
    fig.write_html(save_path)
    # Print a message to the console indicating where the interactive tree has been saved.
    print(f"✅ Interactive glucose tree saved to: {save_path}")
    # Return the save path for further use or confirmation.
    return save_path
