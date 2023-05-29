# sample 10 base points and Print the sampled points
    # for _ in range(10):
    #     x = random.uniform(x_range[0], x_range[1])
    #     y = random.uniform(y_range[0], y_range[1])
    #     base_points.append((x, y))
    # for point in base_points:
    #     print("the base points are:",point)




# training_losses = []
# for epoch in range(num_epochs):
#     for inputs, targets in dataloader:
#         optimizer.zero_grad()
#         outputs = src(inputs)
#         loss = criterion(outputs, targets)
#         training_losses.append(loss.item())
#         loss.backward()
#         optimizer.step()
# plt.plot(training_losses)
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# plt.title('Training Loss')
# plt.show()
# Initializing the list for storing the loss and accuracy


# def find_convexhull(points):
#     # Calculate the convex hull of the points
#     hull = ConvexHull(points)
#
#     # Get the indices of the vertices of the convex hull
#     hull_vertices = hull.vertices
#
#     # Get the coordinates of the vertices of the convex hull
#     hull_points = [points[i] for i in hull_vertices]
#
#     # Calculate the list of edges by connecting consecutive vertices of the convex hull
#     edges = [(hull_points[i], hull_points[(i + 1) % len(hull_points)]) for i in range(len(hull_points))]
#
#     return edges
