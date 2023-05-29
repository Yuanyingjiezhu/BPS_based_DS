import matplotlib.pyplot as plt
import numpy as np

# Define the initial mesh (triangle)
vertices = np.array([
    [-0.2, -0.2],  # Vertex 0
    [0.2, -0.2],  # Vertex 1
    [0, 0.2],  # Vertex 2
])

max_x = np.max(vertices[:, 0])
min_x = np.min(vertices[:, 0])
max_y = np.max(vertices[:, 1])
min_y = np.min(vertices[:, 1])

# Define the square trajectory
num_time_steps = 10  # Number of time steps on each edge
side_length = 0.8  # Side length of the square

# Calculate the x and y displacements for the square trajectory
x_displacement = np.concatenate([
    np.linspace(-side_length / 2, side_length / 2, num_time_steps),  # Bottom edge
    np.ones(num_time_steps) * side_length / 2,  # Right edge
    np.linspace(side_length / 2, -side_length / 2, num_time_steps),  # Top edge
    np.ones(num_time_steps) * -side_length / 2  # Left edge
])
x_displacement = np.clip(x_displacement, -1 - min_x, 1 - max_x)

# print(x_displacement)
y_displacement = np.concatenate([
    np.ones(num_time_steps) * -side_length / 2,  # Bottom edge
    np.linspace(-side_length / 2, side_length / 2, num_time_steps),  # Right edge
    np.ones(num_time_steps) * side_length / 2,  # Top edge
    np.linspace(side_length / 2, -side_length / 2, num_time_steps)  # Left edge
])
y_displacement = np.clip(y_displacement, -1 - min_y, 1 - max_y)
# print(y_displacement)

# Define the rotation speed
rotation_speed = 0  # Fixed rotation speed

# Define the velocity modulation parameters
min_velocity = 1.0   # Minimum velocity
max_velocity = 1.0   # Maximum velocity
velocity_factor = np.linspace(min_velocity, max_velocity, num_time_steps * 4)  # Time-varying scaling factor

# Initialize a list to store the history of mesh transformations
mesh_history = []

# Apply transformations to the vertices at each time step
for i in range(num_time_steps * 4):
    translation = np.array([[x_displacement[i]], [y_displacement[i]], [0]])  # Translation matrix

    # Create the transformation matrix for translation
    transformation_matrix = np.eye(3)
    transformation_matrix[:2, 2] = translation[:2, 0]

    # Apply rotation transformation
    rotation_angle = i * rotation_speed  # Calculate the rotation angle based on the time step
    rotation_matrix = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
        [np.sin(rotation_angle), np.cos(rotation_angle), 0],
        [0, 0, 1]
    ])
    transformation_matrix = np.dot(rotation_matrix, transformation_matrix)

    # Apply velocity modulation to the transformation matrix
    velocity_scaling = np.diag([velocity_factor[i], velocity_factor[i], 1])
    transformation_matrix = np.dot(velocity_scaling, transformation_matrix)

    # Apply the transformation matrix to the vertices
    augmented_vertices = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
    augmented_transformed_vertices = np.dot(augmented_vertices, transformation_matrix.T)
    transformed_vertices = augmented_transformed_vertices[:, 0:2]

    # Append the transformed vertices to the mesh history
    mesh_history.append(transformed_vertices)

# Plot the history of the mesh
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Square Trajectory with Rotation of Triangle Mesh')

for transformed_vertices in mesh_history:
    polygon = plt.Polygon(transformed_vertices[:, :2], edgecolor='blue', fill=None)
    ax.add_patch(polygon)

plt.show()

# Query for a specific time step
query_time_step = 1  # Time step to query

# Retrieve the transformed vertices for the query time step
query_vertices = mesh_history[query_time_step]

# Print the transformed vertices for the query time step
print(f"Vertices at time step {query_time_step}:\n{query_vertices}")
print(np.shape(mesh_history))
