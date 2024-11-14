class RadioStationGraph:
    def __init__(self, stations):
        self.stations = stations
        self.graph = []
        for _ in range(stations):
            self.graph.append([0] * stations)

    def isSafe(self, s, frequency, freq):
        for i in range(self.stations):
            if self.graph[s][i] == 1 and frequency[i] == freq:
                return False
        return True

    def assignFrequenciesUtil(self, num_frequencies, frequency, station):
        if station == self.stations:
            return True
        for freq in range(1, num_frequencies + 1):
            if self.isSafe(station, frequency, freq):
                frequency[station] = freq
                if self.assignFrequenciesUtil(num_frequencies, frequency, station + 1):
                    return True
                frequency[station] = 0
        return False

    def assignFrequencies(self, num_frequencies):
        frequency = [0] * self.stations
        if not self.assignFrequenciesUtil(num_frequencies, frequency, 0):
            print("Solution does not exist")
            return False
        print("Solution exists, and the assigned frequencies are:")
        for f in frequency:
            print(f, end=' ')
        print()
        return True


if __name__ == '__main__':
    stations = int(input("Enter the number of radio stations: "))
    
    # Create a graph
    g = RadioStationGraph(stations)
    print("Enter the adjacency matrix (enter row by row, space-separated):")
    
    for i in range(stations):
        row = list(map(int, input().split()))
        g.graph[i] = row
    
    num_frequencies = int(input("Enter the number of available frequencies: "))
    
    # Assign frequencies
    g.assignFrequencies(num_frequencies)
