class Graph(object):
    """
    Simple graph class for graphs with directed, unweighted or weighted edges.
    Useful for BFS and DFS!
    NOTE: assumes that we use the above SearchNode class for creating edges!
    NOTE: functions are implemented somewhat uniquely so that
    edge weights are only considered if desired. Thus, is still
    useful for uninformed/unweighted graphs (weight is set to 0).
    """
    
    def __init__(self,startNode=None,goalNode=None,weighted = True):
        self.Nodes = set() # tracks unique Nodes
        self.Edges = {} # node:[(child1,cost),(child2,cost)...]
        self.startNode = startNode
        self.goalNode = goalNode
        self.weighted = weighted

    def add_startNode(self,start):
        self.startNode = start

    def add_goalNode(self,goal):
        self.goalNode = goal

    def addNode(self, node):
        if node not in self.Nodes:
            self.Nodes.add(node)
            self.Edges[node] = set()

    def addEdge(self,start,end, directed = False, weight = 0):
        """
        adds an edge between start and end. Creates SearchNode
        object for each if not already existing.
        the parent attribute for end is set to
        """
        if start not in self.Nodes:
            self.addNode(start)
        if end not in self.Nodes:
            self.addNode(end)
        if start in self.Edges:
            prev = self.Edges[start]
            prev.add((end,weight))
            self.Edges[start] = prev
        else:
            self.Edges[start] = set((end,weight))
        if not directed:
            if end in self.Edges:
                prev = self.Edges[end]
                prev.add((start,weight))
                self.Edges[end] = prev
            else:
                self.Edges[end] = set((start,weight))
            

    def getStartNode(self):
        return self.startNode
    
    def getGoalNode(self):
        return self.goalNode
    
    def getChildren(self, node):
        # only returns a list of children, not their weights
        # thus, still nice to use with BFS, etc
        # use .getWeight to get weights of a given edge
        out = [child[0] for child in self.Edges[node]]
        return out[:]
    
    def getNodes(self):
        return self.Nodes.copy()
    
    def getAllEdges(self):
        return self.Edges.copy()
    
    def getEdgeWeight(self, start,end):
        # get an edge weight
        try:
            children = self.Edges[start]
            for child in children:
                if child[0] == end:
                    return child[1]
        except:
            raise

    def isWeighted(self):
        return self.weighted