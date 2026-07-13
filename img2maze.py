import cv2
from PIL import Image
import numpy as np
import sys
import os
import networkx as nx
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

SIZE = 127
BORDER = 5
sys.setrecursionlimit(10000)


def process_image(im):
    """
    Helper function that processes image into a 127x127 edge-detected greyscale image
    """
    im = im.convert("L")
    im_arr = np.asarray(im)
    # find edges in image
    edges = cv2.Canny(im_arr, 100, 200)
    # resize image
    resized_im = cv2.resize(
        edges,
        dsize=(SIZE - (BORDER * 2), SIZE - (BORDER * 2)),
        interpolation=cv2.INTER_AREA,
    )
    border_im = cv2.copyMakeBorder(
        resized_im,
        BORDER,
        BORDER,
        BORDER,
        BORDER,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    return border_im


def init_graph(im):
    """
    Helper function that creates a graph with a node (room) for every pixel and edges to indicate walls between rooms
    """
    global G, max_fill
    G = nx.Graph()
    # add nodes and edges
    nodes_to_add = []
    edges_to_add = []
    max_fill = -1
    for i in range(SIZE * SIZE):
        row = i // SIZE
        col = i % SIZE
        max_fill = max(max_fill, im[(SIZE - 1) - row, col])
        node = (
            i,
            {
                "id": i,
                "row": row,
                "col": col,
                "visited": False,
                "fill": im[(SIZE - 1) - row, col],
            },
        )
        nodes_to_add.append(node)

        # handle boundary conditions
        edge_e = (i, i + 1) if col < SIZE - 1 else None
        edge_w = (i, i - 1) if col > 0 else None
        edge_n = (i, i - SIZE) if row > 0 else None
        edge_s = (i, i + SIZE) if row < SIZE - 1 else None
        edges_to_add.extend(
            list(filter(lambda a: a != None, [edge_e, edge_w, edge_n, edge_s]))
        )

    G.add_nodes_from(nodes_to_add)
    G.add_edges_from(edges_to_add)


def get_neighbours(node):
    """
    Helper function that returns a list of neighbours for a given node
    """
    neighbour_n = node["id"] - SIZE if node["row"] > 0 else None
    neighbour_s = node["id"] + SIZE if node["row"] < SIZE - 1 else None
    neighbour_e = node["id"] + 1 if node["col"] < SIZE - 1 else None
    neighbour_w = node["id"] - 1 if node["col"] > 0 else None
    return list(
        filter(
            lambda a: a != None, [neighbour_n, neighbour_s, neighbour_e, neighbour_w]
        )
    )


def generate_maze(node, im):
    """
    Helper function that generates a maze using DFS from the given graph
    """
    node["visited"] = True
    neighbours = get_neighbours(node)
    random.shuffle(neighbours)
    for neighbour in neighbours:
        neighbour_node = G.nodes[neighbour]
        if (
            not neighbour_node["visited"]
            and im[(SIZE - 1) - neighbour_node["row"], neighbour_node["col"]] == 0
        ):
            G.remove_edge(node["id"], neighbour_node["id"])
            generate_maze(neighbour_node, im)


def visualize_maze(cell_size):
    """
    Helper function that visualizes a generated maze
    """
    fig = plt.figure(figsize=(24, 24))
    ax = plt.axes()
    ax.set_aspect("equal")
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    # plot boundaries
    ax.plot([0, SIZE * cell_size], [0, 0], color="k")
    ax.plot([0, SIZE * cell_size], [SIZE * cell_size, SIZE * cell_size], color="k")
    ax.plot([0, 0], [0, (SIZE - 1) * cell_size], color="k")
    ax.plot(
        [SIZE * cell_size, SIZE * cell_size], [cell_size, SIZE * cell_size], color="k"
    )
    for edge in list(G.edges):
        node = G.nodes[edge[0]]
        neighbour = G.nodes[edge[1]]
        # plot walls
        if node["row"] == neighbour["row"]:
            x = neighbour["col"] * cell_size
            y_1 = node["row"] * cell_size
            y_2 = (node["row"] + 1) * cell_size
            ax.plot([x, x], [y_1, y_2], color="k")
        elif node["col"] == neighbour["col"]:
            y = neighbour["row"] * cell_size
            x_1 = node["col"] * cell_size
            x_2 = (node["col"] + 1) * cell_size
            ax.plot([x_1, x_2], [y, y], color="k")
        # fill original image
        if node["fill"] > 0:
            x = node["col"] * cell_size
            y = node["row"] * cell_size
            rect = patches.Rectangle(
                (x, y), cell_size, cell_size, color=str(node["fill"] / max_fill)
            )
            ax.add_patch(rect)
            node["fill"] = 0
        elif neighbour["fill"] > 0:
            x = neighbour["col"] * cell_size
            y = neighbour["row"] * cell_size
            rect = patches.Rectangle(
                (x, y), cell_size, cell_size, color=str(neighbour["fill"] / max_fill)
            )
            ax.add_patch(rect)
            neighbour["fill"] = 0


if __name__ == "__main__":
    print("Starting image to maze conversion...")
    if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]):
        sys.exit("Please imput a valid path to the image you would like to convert.")
    im = Image.open(sys.argv[1])
    processed_im = process_image(im)
    init_graph(processed_im)
    generate_maze(G.nodes[0], processed_im)
    maze = visualize_maze(cell_size=30)
    plt.savefig(
        "mazes/{}{}.png".format(sys.argv[1][:-4].removeprefix("imgs/"), "_maze")
    )
    print("Maze generation complete!")
