#include <algorithm>
#include <cstddef>
#include <iostream>
#include <vector>

using namespace std;

template <typename T> void print_vec(const vector<T> &vec) {
  for (T x : vec)
    cout << x << " ";
  cout << "\n";
}

// XXX: Merges two sorted lists in order and in place.
template <typename T> void merge(const vector<T> &r, vector<T> &res) {
  size_t start_pos = 0;
  for (size_t i = 0; i < res.size(); ++i) {
    size_t les = count_if(r.begin() + start_pos, r.end(),
                          [&res, &i](T x) { return x <= res[i]; });
    // XXX: Now add the elements from j from start_pos until the les to
    // res then add first element of l
    for (size_t j = start_pos; j < start_pos + les; ++j) {
      res.insert(res.begin() + i, r[j]);
      i += 1; // Increment i after insertion
    }
    start_pos += les;
  }
  // XXX: Finally, if there are elements remaining in the r vector, just
  // copy them
  for (size_t k = start_pos; k < r.size(); ++k)
    res.push_back(r[k]);
}

// XXX: Given the k sorted vectors; merge them one after another.
template <typename T> void merge_k(const vector<vector<T>> kl, vector<T> &res) {
  for (T x : kl[0])
    res.push_back(x);
  // XXX: Space complexity is now O(|kl|*|kl|)
  for (size_t i = 1; i < kl.size(); ++i) {
    merge(kl[i], res);
  }
}

int main() {
  vector<vector<int>> kl{{1, 2, 80000}, {0, 1, 5, 100, 9000}, {1, 45, 90}};
  vector<int> res;
  merge_k(kl, res);
  for (int x : res) {
    cout << x << " ";
  }
  cout << "\n";
  return 0;
}
