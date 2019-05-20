"""
Cluster data.
"""


from collections import Counter, defaultdict, deque
import matplotlib.pyplot as plt
import networkx as nx
import pickle



def get_user_info(name):
    """
    load stored user info, list of dicts
    (screen_name, id, friends_id)
    """
    
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def print_num_friends(users):
    """Print the number of friends per candidate, sorted by candidate name.
    See Log.txt for an example.
    Args:
        users....The list of user dicts.
    Returns:
        Nothing
    """
    ###TODO
    users = sorted(users, key=lambda x: x['screen_name'])
    for user in users:
        print('%s %d' % (user['screen_name'], len(user['friends'])))
    return


def count_friends(users):
    """ Count how often each friend is followed.
    Args:
        users: a list of user dicts
    Returns:
        a Counter object mapping each friend to the number of candidates who follow them.
        Counter documentation: https://docs.python.org/dev/library/collections.html#collections.Counter

    In this example, friend '2' is followed by three different users.
    >>> c = count_friends([{'friends': [1,2]}, {'friends': [2,3]}, {'friends': [2,3]}])
    >>> c.most_common()
    [(2, 3), (3, 2), (1, 1)]
    """
    ###TODO
    count = Counter()
    for user in users:
        count.update(user['friends'])
    count.most_common()
    return count


def friend_overlap(users):
    """
    Compute the number of shared accounts followed by each pair of users.

    Args:
        users...The list of user dicts.

    Return: A list of tuples containing (user1, user2, N), where N is the
        number of accounts that both user1 and user2 follow.  This list should
        be sorted in descending order of N. Ties are broken first by user1's
        screen_name, then by user2's screen_name (sorted in ascending
        alphabetical order). See Python's builtin sorted method.

    In this example, users 'a' and 'c' follow the same 3 accounts:
    >>> friend_overlap([
    ...     {'screen_name': 'a', 'friends': ['1', '2', '3']},
    ...     {'screen_name': 'b', 'friends': ['2', '3', '4']},
    ...     {'screen_name': 'c', 'friends': ['1', '2', '3']},
    ...     ])
    [('a', 'c', 3), ('a', 'b', 2), ('b', 'c', 2)]
    """
    ###TODO
    overlap = []
    for i in range(len(users)-1):
        friends = users[i]['friends']
        for j in range(i+1, len(users)):
            n = 0
            friends2 = users[j]['friends']
            for f in friends:
                if f in friends2:
                    n = n+1
            overlap.append(
                (users[i]['screen_name'], users[j]['screen_name'], n))
    overlap = sorted(overlap, key=lambda x: -x[2])
    return overlap


def create_graph(users, friend_counts, min_common):
    """ Create a networkx undirected Graph, adding each candidate and friend
        as a node.  Note: while all candidates should be added to the graph,
        only add friends to the graph if they are followed by more than one
        candidate. (This is to reduce clutter.)

        Each candidate in the Graph will be represented by their screen_name,
        while each friend will be represented by their user id.

    Args:
      users...........The list of user dicts.
      friend_counts...The Counter dict mapping each friend to the number of candidates that follow them.
    Returns:
      A networkx Graph
    """

    #list of friends ids followed by more than min_common users
    friends = [x for x in friend_counts if friend_counts[x] > min_common]
    
    #draw undirected graph
    graph = nx.Graph()
    #add nodes for friends
    for x in friends:
        graph.add_node(x)
    #add users nodes
    for user in users:
        graph.add_node(user['id'])
        #list of friends should be plotted
        fndlst = set(user['friends']) & set(friends)
        #add edges for each node
        for fnd in fndlst:
            graph.add_edge(fnd, user['id'])

    nx.draw(graph, with_labels=True)
    
    return graph


def draw_network(graph, users, filename):
    """
    Draw the network to a file.
    Only label the candidate nodes; the friend
    nodes should have no labels (to reduce clutter).

    params:
        graph...........the undirected graph created
        users...........list of dicts
        filename........the name of the saved network figure
    """

    #only users have lables
    label = {}
    for n in graph.nodes():
        for u in users:
            if n in u['id']:
                label[n] = u['screen_name']
    
    plt.figure(figsize=(15, 15))
    plt.axis('off')

    nx.draw_networkx(graph, labels=label, alpha=.5, node_size=100, width=.5)
    plt.savefig(filename)


def get_subgraph(graph, min_degree):
    """Return a subgraph containing nodes whose degree is
    greater than or equal to min_degree.
    We'll use this in the main method to prune the original graph.

    Params:
      graph........a networkx graph
      min_degree...degree threshold
    Returns:
      a networkx graph, filtered as defined above.

    >>> subgraph = get_subgraph(example_graph(), 3)
    >>> sorted(subgraph.nodes())
    ['B', 'D', 'F']
    >>> len(subgraph.edges())
    2
    """
    ###TODO
    subgraph = graph.copy()
    degrees = nx.degree(graph)

    for key in degrees:
        if key[1] < min_degree and key[0] in graph.nodes():
          subgraph.remove_node(key[0])
    return subgraph


def bfs(graph, root, max_depth):
    """
    Perform breadth-first search to compute the shortest paths from a root node to all
    other nodes in the graph. To reduce running time, the max_depth parameter ends
    the search after the specified depth.
    E.g., if max_depth=2, only paths of length 2 or less will be considered.
    This means that nodes greather than max_depth distance from the root will not
    appear in the result.

    You may use these two classes to help with this implementation:
      https://docs.python.org/3.5/library/collections.html#collections.defaultdict
      https://docs.python.org/3.5/library/collections.html#collections.deque

    Params:
      graph.......A networkx Graph
      root........The root node in the search graph (a string). We are computing
                  shortest paths from this node to all others.
      max_depth...An integer representing the maximum depth to search.

    Returns:
      node2distances...dict from each node to the length of the shortest path from
                       the root node
      node2num_paths...dict from each node to the number of shortest paths from the
                       root node to this node.
      node2parents.....dict from each node to the list of its parents in the search
                       tree

    In the doctests below, we first try with max_depth=5, then max_depth=2.

    >>> node2distances, node2num_paths, node2parents = bfs(example_graph(), 'E', 5)
    >>> sorted(node2distances.items())
    [('A', 3), ('B', 2), ('C', 3), ('D', 1), ('E', 0), ('F', 1), ('G', 2)]
    >>> sorted(node2num_paths.items())
    [('A', 1), ('B', 1), ('C', 1), ('D', 1), ('E', 1), ('F', 1), ('G', 2)]
    >>> sorted((node, sorted(parents)) for node, parents in node2parents.items())
    [('A', ['B']), ('B', ['D']), ('C', ['B']), ('D', ['E']), ('F', ['E']), ('G', ['D', 'F'])]
    >>> node2distances, node2num_paths, node2parents = bfs(example_graph(), 'E', 2)
    >>> sorted(node2distances.items())
    [('B', 2), ('D', 1), ('E', 0), ('F', 1), ('G', 2)]
    >>> sorted(node2num_paths.items())
    [('B', 1), ('D', 1), ('E', 1), ('F', 1), ('G', 2)]
    >>> sorted((node, sorted(parents)) for node, parents in node2parents.items())
    [('B', ['D']), ('D', ['E']), ('F', ['E']), ('G', ['D', 'F'])]
    """
    q = deque()
    q.append(root)
    seen = set()
    seen.add(root)
    node2distances = defaultdict(int)
    node2num_paths = defaultdict(int)
    node2num_paths[root] = 1
    node2parents = defaultdict(list)
    #recursive BFS
    while q:
        n = q.popleft()
        #jump out when one search reach the max_depth
        if node2distances[n] == max_depth:
            continue
        for nn in graph.neighbors(n):
            if nn not in seen:
                q.append(nn)
                node2distances[nn] = node2distances[n] + 1
                node2num_paths[nn] = node2num_paths[n]
                node2parents[nn].append(n)
                seen.add(nn)
            elif node2distances[nn] == node2distances[n] + 1:
                node2parents[nn].append(n)
                node2num_paths[nn] += node2num_paths[n]
    return node2distances, node2num_paths, node2parents


def bottom_up(root, node2distances, node2num_paths, node2parents):
    """
    Compute the final step of the Girvan-Newman algorithm.
    See p 352 From your text:
    https://github.com/iit-cs579/main/blob/master/read/lru-10.pdf
        The third and final step is to calculate for each edge e the sum
        over all nodes Y of the fraction of shortest paths from the root
        X to Y that go through e. This calculation involves computing this
        sum for both nodes and edges, from the bottom. Each node other
        than the root is given a credit of 1, representing the shortest
        path to that node. This credit may be divided among nodes and
        edges above, since there could be several different shortest paths
        to the node. The rules for the calculation are as follows: ...

    Params:
      root.............The root node in the search graph (a string). We are computing
                       shortest paths from this node to all others.
      node2distances...dict from each node to the length of the shortest path from
                       the root node
      node2num_paths...dict from each node to the number of shortest paths from the
                       root node that pass through this node.
      node2parents.....dict from each node to the list of its parents in the search
                       tree
    Returns:
      A dict mapping edges to credit value. Each key is a tuple of two strings
      representing an edge (e.g., ('A', 'B')). Make sure each of these tuples
      are sorted alphabetically (so, it's ('A', 'B'), not ('B', 'A')).

      Any edges excluded from the results in bfs should also be exluded here.

    >>> node2distances, node2num_paths, node2parents = bfs(example_graph(), 'E', 5)
    >>> result = bottom_up('E', node2distances, node2num_paths, node2parents)
    >>> sorted(result.items())
    [(('A', 'B'), 1.0), (('B', 'C'), 1.0), (('B', 'D'), 3.0), (('D', 'E'), 4.5), (('D', 'G'), 0.5), (('E', 'F'), 1.5), (('F', 'G'), 0.5)]
    """
    values = defaultdict(list)
    bottom_up = defaultdict(list)
    max_level = max(node2distances.values())

    for node, level in node2distances.items():
        values[node] = 1
    for distance in range(max_level, 0, -1):
        for node, level in node2distances.items():
            if level == distance:
                if node2num_paths[node] > 1:
                        values[node] = values[node]/node2num_paths[node]
                for child in node2parents[node]:
                    if child not in values:
                        values[child] = values[node]
                    else:
                        values[child] = values[child] + values[node]
                    aux = [node, child]
                    aux.sort()
                    bottom_up[aux[0], aux[1]] = values[node]
    return bottom_up


def approximate_betweenness(graph, max_depth):
    """
    Compute the approximate betweenness of each edge, using max_depth to reduce
    computation time in breadth-first search.

    You should call the bfs and bottom_up functions defined above for each node
    in the graph, and sum together the results. Be sure to divide by 2 at the
    end to get the final betweenness.

    Params:
      graph.......A networkx Graph
      max_depth...An integer representing the maximum depth to search.

    Returns:
      A dict mapping edges to betweenness. Each key is a tuple of two strings
      representing an edge (e.g., ('A', 'B')). Make sure each of these tuples
      are sorted alphabetically (so, it's ('A', 'B'), not ('B', 'A')).

    >>> sorted(approximate_betweenness(example_graph(), 2).items())
    [(('A', 'B'), 2.0), (('A', 'C'), 1.0), (('B', 'C'), 2.0), (('B', 'D'), 6.0), (('D', 'E'), 2.5), (('D', 'F'), 2.0), (('D', 'G'), 2.5), (('E', 'F'), 1.5), (('F', 'G'), 1.5)]
    """
    bot = defaultdict(list)
    betweenness = defaultdict(list)
    for node in graph.nodes():
        node2distances, node2num_paths, node2parents = bfs(graph, node, max_depth)
        bot = bottom_up(node, node2distances, node2num_paths, node2parents)
        for i, j in bot:
            if (i, j) in betweenness:
                betweenness[(i, j)] = betweenness[(i, j)] + bot[(i, j)]
            else:
                betweenness[(i, j)] = bot[(i, j)]

    for key in betweenness:
        betweenness[key] = betweenness[key]/2
    return betweenness


def partition_girvan_newman(graph, max_depth, num_clusters):
    """
    Use the approximate_betweenness implementation to partition a graph.
    Unlike in class, here you will not implement this recursively. Instead,
    just remove edges until more than one component is created, then return
    those components.
    That is, compute the approximate betweenness of all edges, and remove
    them until multiple comonents are created.

    Note: the original graph variable should not be modified. Instead,
    make a copy of the original graph prior to removing edges.

    Params:
      graph..........A networkx Graph created before
      max_depth......An integer representing the maximum depth to search.
      num_clusters...number of clusters want

    Returns:
      clusters...........A list of networkx Graph objects, one per partition.
      users_graph........the partitioned users graph.
    """
 
    clusters = []

    partition_edge = list(sorted(approximate_betweenness(graph, max_depth).items(), key=lambda x:(-x[1], x[0])))
    
    for i in range(0, len(partition_edge)):
        graph.remove_edge(*partition_edge[i][0])
        clusters = list(nx.connected_component_subgraphs(graph))
        if len(clusters) >= num_clusters:
            break

    #remove outliers
    new_clusters = [cluster for cluster in clusters if len(cluster.nodes()) > 1]

    return new_clusters, graph


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


def main():
    users = get_user_info('user_info')
    print("Got users info.")
    print('Number of friends of each user:', print_num_friends(users))
    
    friend_counts = count_friends(users)
    print('Most common friends:\n%s' % str(friend_counts.most_common(5)))
    print('Friend Overlap:\n%s' % str(friend_overlap(users)))
    
    graph = create_graph(users, friend_counts, 0)
    print('graph has %s nodes and %s edges' % (len(graph.nodes()), len(graph.edges())))
    draw_network(graph, users, 'original_net.png')
    print('network drawn to file original_net.png')

    subgraph = create_graph(users, friend_counts, 1)
    print('subgraph has %s nodes and %s edges' % (len(subgraph.nodes()), len(subgraph.edges())))
    draw_network(subgraph, users, 'pruned_net.png')
    print('network drawn to file pruned_net.png')

    clusters, partitioned_graph = partition_girvan_newman(subgraph,5, 3)
    save_obj(clusters, 'clusters')

    print(len(clusters))
    print('cluster 1 has %d nodes, cluster 2 has %d nodes, cluster 3 has %d nodes' %(len(clusters[0].nodes()), len(clusters[1].nodes()), len(clusters[2].nodes())))

    draw_network(partitioned_graph, users, 'clusters_net.png')
    print('network drawn to file clusters_net.png')


if __name__ == '__main__':
    main()
