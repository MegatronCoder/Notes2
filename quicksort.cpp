// multithreaded quicksort

#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <omp.h>

using namespace std;

class QuickSortMultiThreading {
public:
	QuickSortMultiThreading(int start, int end, vector<int>& arr) 
		: start_(start), end_(end), arr_(arr) {}
	
	// Finding random pivoted and partition
	// array on a pivot.
	// There are many different
	// partitioning algorithms.
	int partition(int start, int end, vector<int>& arr) {
		int i = start;
		int j = end;

		// Decide random pivot
		int pivoted = rand() % (j - i + 1) + i;

		// Swap the pivoted with end
		// element of array
		int t = arr[j];
		arr[j] = arr[pivoted];
		arr[pivoted] = t;
		j--;

		// Start partitioning
		while (i <= j) {
			if (arr[i] <= arr[end]) {
				i++;
				continue;
			}
			if (arr[j] >= arr[end]) {
				j--;
				continue;
			}
			t = arr[j];
			arr[j] = arr[i];
			arr[i] = t;
			j--;
			i++;
		}

		// Swap pivoted to its
		// correct position
		t = arr[j + 1];
		arr[j + 1] = arr[end];
		arr[end] = t;
		return j + 1;
	}

	// Function to implement
	// QuickSort method
	void operator() () {
		// Base case
		if (start_ >= end_) {
			return;
		}

		// Find partition
		int p = partition(start_, end_, arr_);

		// Divide array
		QuickSortMultiThreading left(start_, p - 1, arr_);
		QuickSortMultiThreading right(p + 1, end_, arr_);

		// Left subproblem as separate thread
		#pragma omp parallel sections
		{
			#pragma omp section
			{
				left();
			}
			#pragma omp section
			{
				right();
			}
		}
	}

private:
	int start_;
	int end_;
	vector<int>& arr_;
};

int main() {
	int n = 7;
	vector<int> arr = {54, 64, 95, 82, 12, 32, 63};
	srand(time(NULL));
	QuickSortMultiThreading(0, n - 1, arr)();
	for (int i = 0; i < n; i++) {

		// Print sorted elements
		cout << arr[i] << " ";
	}
	cout << endl;
	return 0;
}
