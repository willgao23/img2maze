import cv2
from PIL import Image
import numpy as np
import sys
import os
import networkx as nx
import random

SIZE = 127


def process_image(im):
    """
    Helper function that processes image into a 127x127 edge-detected greyscale image
    """
    im = im.convert("L")
    im_arr = np.asarray(im)
    # find edges in image
    edges = cv2.Canny(im_arr, 100, 200)
    # resize image
    resized_im = cv2.resize(edges, dsize=(SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return resized_im


def init_graph(im):
    """
    Helper function that creates a graph with a node for every pixel
    """
    global G
    G = nx.Graph()
    # add nodes and edges
    nodes_to_add = []
    for i in range(im.shape[0] * im.shape[1]):
        row = i // im.shape[0]
        col = i % im.shape[1]
        node = (i, {"id": i, "row": row, "col": col, "visited": False})
        nodes_to_add.append(node)
    G.add_nodes_from(nodes_to_add)


def get_neighbours(node):
    """
    Helper function that returns a list of neighbours for a given node
    """
    neighbour_n = node["id"] - SIZE if node["row"] > 0 else None
    neighbour_s = node["id"] + SIZE if node["row"] < SIZE - 1 else None
    neighbour_e = node["id"] + 1 if node["col"] < SIZE - 1 else None
    neighbour_w = node["id"] - 1 if node["col"] > 0 else None
    return filter(
        lambda a: a != None, [neighbour_n, neighbour_s, neighbour_e, neighbour_w]
    )


def generate_maze(node, im):
    """
    Helper function that generates a maze using DFS from the given graph
    """
    node["visited"] = True
    neighbours = random.shuffle(get_neighbours(node))
    for neighbour in neighbours:
        neighbour_node = G.nodes[neighbour]
        if (
            not neighbour_node["visited"]
            and im[neighbour_node["row"], neighbour_node["col"]] == 0
        ):
            G.add_edge((node["id"], neighbour_node["id"]))
            generate_maze(neighbour_node, im)


if __name__ == "__main__":
    print("Starting image to maze conversion...")
    if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]):
        sys.exit("Please imput a valid path to the image you would like to convert.")
    im = Image.open(sys.argv[1])
    processed_im = process_image(im)
    init_graph(processed_im)
    generate_maze(G.nodes[0], im)
