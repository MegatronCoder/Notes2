class Bellman_Ford:
    def __init__(self, vertices):
        self.vertices = vertices
        self.graph = []
    
    def add_edge(self, u, v, w):
        self.graph.append([u, v, w])
    
    def Bellman(self, src):
        dist = [10000]*self.vertices
        dist[src] = 0
        
        for _ in range(self.vertices - 1):
            for u, v, w in self.graph:
                if dist[u] != 10000 and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
        
        for u, v, w in self.graph:
            if dist[u] != 10000 and dist[u] + w < dist[v]:
                print("Graph contains Negative Weight Cycle!")
                
        self.printArr(dist)
    
    def printArr(self, dist):
        print("Weights according to all edges:- ")
        for i in range(self.vertices):
            print(f"{i} :- {dist[i]}")

if __name__ == "__main__":
    vertices = int(input("Enter number of vertices/nodes in the Graph :- "))
    edges = int(input("How many edges does the graph consists :- "))
    print("Enter edge in the (Start End Weight) format :- ")
    g = Bellman_Ford(vertices)
    
    for i in range(edges):
        a, b, w = input().split(" ")
        g.add_edge(int(a), int(b), int(w))
    
    g.Bellman(0)
