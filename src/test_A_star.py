
import numpy as np
import heapq

nodes = [[0,0,100,100,-1],
         [0,-1,-1,-1,-1],
         [0,0,0,-1,100],
         [-1,-1,-1,0,100],
         [-1,-1,-1,-1,0]]

def eucDist(coord1,coord2): # coord = (x,y)
        return np.sqrt((coord1[0]-coord2[0])**2+(coord1[1]+coord1[1])**2)

def AStarWithExpandedList():
        
        nodes = [[0,0,100,100,-1],
                [0,-1,-1,-1,-1],
                [0,0,0,-1,100],
                [-1,-1,-1,0,100],
                [-1,-1,-1,-1,0]]
        
        def computeH(u,v): 
            goal = (4,4)
            return eucDist((u,v),goal)
        
        def getChildren(i,j):
             # coord =(u,v)
             return [(i+1,j),(i-1,j),(i,j+1),(i,j-1),(i+1,j+1),(i-1,j-1),(i+1,j-1),(i-1,j+1)]

        """
        A* with an expanded list, assumes consistent heuristic
        """
        S = (0,0)
        G = (4,4)

        expanded = set()
        Q = [(computeH(S[0],S[1]),(0,[S]))] # (cost_to_come+cost_incurred,(cost_incurred, [partial path]))
        heapq.heapify(Q)

        while Q:
            #print(Q)
            shortest = heapq.heappop(Q)
            f , N = shortest
            partialPath = N[1]
            costIncurred = N[0]
            head = partialPath[-1]
            if head == G:
                return costIncurred, partialPath
            elif head in expanded:
                continue
            else:
                expanded.add(head) # NOTE: check!
                children = getChildren(head[0],head[1])
                for child in children:
                    if child not in expanded:
                        try:
                            val = nodes[child[0]][child[1]]
                            if val == 0:
                                extension = partialPath + [child]
                                costToChild = eucDist(head,child) + costIncurred
                                heapq.heappush(Q,(costToChild+computeH(child[0],child[1]),(costToChild,extension)))
                        except: # out of bounds
                             continue 
        return None

print(AStarWithExpandedList())