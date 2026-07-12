import cv2
from PIL import Image
import numpy as np
import sys
import os
import networkx as nx


def process_image(im):
    """
    Helper function that processes image into a 127x127 edge-detected greyscale image
    """
    im = im.convert("L")
    im_arr = np.asarray(im)
    # find edges in image
    edges = cv2.Canny(im_arr, 100, 200)
    # resize image
    resized_im = cv2.resize(edges, dsize=(127, 127), interpolation=cv2.INTER_AREA)
    return resized_im


def init_graph(im):
    """
    Helper function that creates a graph with a node (room) for every pixel and edges to indicate
    open or traversable paths between rooms
    """
    G = nx.Graph()
    # add nodes and edges
    nodes_to_add = []
    edges_to_add = []
    for i in range(im.shape[0] * im.shape[1]):
        row = i // im.shape[0]
        col = i % im.shape[1]
        node = (i, {"row": i // im.shape[0], "col": i % im.shape[1]})
        nodes_to_add.append(node)

        # check if pixel in original image is an edge, ignore paths to other rooms if so
        if im[row, col] > 0:
            continue

        # handle boundary conditions
        edge_e = (i, i + 1) if col < im.shape[1] - 1 else None
        edge_w = (i, i - 1) if col > 0 else None
        edge_n = (i, i - im.shape[1]) if row > 0 else None
        edge_s = (i, i + im.shape[1]) if row < im.shape[0] - 1 else None
        edges_to_add.extend(
            filter(lambda a: a != None, [edge_e, edge_w, edge_n, edge_s])
        )

    G.add_nodes_from(nodes_to_add)
    G.add_edges_from(edges_to_add)
    return


if __name__ == "__main__":
    print("Starting image to maze conversion...")
    if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]):
        sys.exit("Please imput a valid path to the image you would like to convert.")
    im = Image.open(sys.argv[1])
    processed_im = process_image(im)
    G = init_graph(processed_im)
