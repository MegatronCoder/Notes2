import heapq
import copy

class Node:
    def __init__(self, x, y, assigned, parent):
        self.parent = parent
        self.pathCost = 0
        self.cost = 0
        self.workerID = x
        self.jobID = y
        self.assigned = copy.deepcopy(assigned)
        if y != -1:
            self.assigned[y] = True

class CustomHeap:
    def __init__(self):
        self.heap = []

    def push(self, node):
        heapq.heappush(self.heap, (node.cost, node))

    def pop(self):
        if self.heap:
            _, node = heapq.heappop(self.heap)
            return node
        return None

def new_node(x, y, assigned, parent):
    return Node(x, y, assigned, parent)

def calculate_cost(cost_matrix, x, y, assigned, n):
    cost = 0
    available = [True] * n
    for i in range(x + 1, n):
        min_val, min_index = float('inf'), -1
        for j in range(n):
            if not assigned[j] and available[j] and cost_matrix[i][j] < min_val:
                min_index = j
                min_val = cost_matrix[i][j]
        cost += min_val
        available[min_index] = False
    return cost

def print_assignments(min_node):
    if min_node.parent is None:
        return
    print_assignments(min_node.parent)
    print(f"Worker {min_node.workerID} is assigned to Job {min_node.jobID}")

def find_min_cost(cost_matrix, n):
    pq = CustomHeap()
    assigned = [False] * n
    root = new_node(-1, -1, assigned, None)
    root.pathCost = root.cost = 0
    root.workerID = -1
    pq.push(root)
    
    while True:
        min_node = pq.pop()
        i = min_node.workerID + 1
        if i == n:
            print_assignments(min_node)
            return min_node.cost
        for j in range(n):
            if not min_node.assigned[j]:
                child = new_node(i, j, min_node.assigned, min_node)
                child.pathCost = min_node.pathCost + cost_matrix[i][j]
                child.cost = child.pathCost + calculate_cost(cost_matrix, i, j, child.assigned, n)
                pq.push(child)

if __name__ == "__main__":
    # Get the size of the matrix (n)
    n = int(input("Enter the number of workers/jobs (n): "))
    
    # Initialize an empty cost matrix
    cost_matrix = []
    
    # Get the cost matrix input from the user
    print("Enter the cost matrix values row by row:")
    for i in range(n):
        row = list(map(int, input(f"Row {i+1}: ").split()))
        if len(row) != n:
            print(f"Error: Row must contain exactly {n} values.")
            exit(1)
        cost_matrix.append(row)
    
    # Call the function to find the minimum cost
    optimal_cost = find_min_cost(cost_matrix, n)
    if optimal_cost is not None:
        print("Optimal Cost:", optimal_cost)
    else:
        print("No solution found.")
