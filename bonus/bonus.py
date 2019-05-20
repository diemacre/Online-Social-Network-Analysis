import networkx as nx

def jaccard_wt(graph, node):
    """
    The weighted jaccard score, defined above.
    Args:
      graph....a networkx graph
      node.....a node to score potential new edges for.
    Returns:
      A list of ((node, ni), score) tuples, representing the 
                score assigned to edge (node, ni)
                (note the edge order)
    """
    neigh = set(graph.neighbors(node))
    scores = []
    for n in graph.nodes():
        if (n not in neigh) and (n != node):
            neigh2 = set(graph.neighbors(n))
            nodeScore = 0
            Adeg = 0
            Bdeg = 0
            for i in neigh:
                Adeg += len(set(graph.neighbors(i)))
            for j in neigh2:
                Bdeg += len(set(graph.neighbors(j)))
            for common in neigh:
                if common  in neigh2:
                    nodeScore += 1 / len(set(graph.neighbors(common)))
            scores.append(((node, n), nodeScore / ((1 / Adeg) + (1 / Bdeg))))
    return sorted(scores, key=lambda x: x[1])

 
'''
def example_graph():
    """
    Create the example graph from class. Used for testing.
    Do not modify.
    """
    g = nx.Graph()
    g.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'), ('D', 'E'), ('D', 'F'), ('D', 'G'), ('E', 'F'), ('G', 'F')])
    return g

def main():
    g = example_graph()
    print(jaccard_wt(g, 'G'))

if __name__ == '__main__':
    main()
'''