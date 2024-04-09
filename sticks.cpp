#include <algorithm>
#include <assert.h>
#include <bitset>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <ostream>
#include <queue>
#include <random>
#include <ranges>
#include <set>
#include <stack>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#define st size_t

using namespace std;

// XXX: Returns the iterator to the value if found else end.
// XXX: Obviously we assume that the list is sorted.
template <class T, typename U>
enable_if<is_integral<U>::value, T>::type binary_find(T s, T e, const U v) {
  T m;
  while (e - s > 2) {
    st half = (e - s) / 2;
    // cout << "half: " << half << "\n";
    // cout << "size: " << (e - s) << "\n";
    m = (s + half);
    // cout << "m: " << *m << " v is: " << v << "\n";
    // cout << "v <= *m: " << (v <= *m) << "\n";
    if (v <= *m) {
      // XXX: Move the end to m+1 position
      // cout << "moving end pointer\n";
      e = m + 1;
    } else {
      // XXX: Move the start to m+1 position;
      // cout << "moving start pointer\n";
      s = m + 1;
    }
  }
  // XXX: This final one should be O(2) always
  while (s != e) {
    if (*s == v)
      break;
    ++s;
  }
  return s;
}

template <typename T> ostream &operator<<(ostream &os, const vector<T> &vec) {
  for (const T &v : vec)
    cout << v << " ";
  cout << "\n";
  return os;
}

// XXX: Can print anything via iterators. Can also print value of arrays
// via pointers.
void print_iter(auto f, auto l) {
  st counter = 0;
  while (f != l) {
    cout << counter << ": " << *f << " ";
    ++f;
    ++counter;
  }
  cout << "\n";
}

template <typename T> ostream &operator<<(ostream &os, const pair<T, T> &vec) {
  os << "<" << vec.first << "," << vec.second << ">\n";
  return os;
}

template <typename T>
ostream &operator<<(ostream &os, const tuple<T, T, T> &vec) {
  os << "<" << get<0>(vec) << "," << get<1>(vec) << "," << get<2>(vec) << ">\n";
  return os;
}

template <typename T>
void print_vec(const vector<T> &vec, const bool b = true) {
  for (const T &v : vec) {
    cout << v << " ";
  }
  if (b)
    cout << "\n";
  else
    cout << " ";
}

template <typename K, typename V> void print_map(unordered_map<K, V> m) {
  for (auto &[k, v] : m) {
    cout << k << ": " << v << "\n";
  }
}

void print_multiset(const multiset<tuple<int, int, int>> &v) {
  for (auto const &p : v) {
    cout << "<" << get<0>(p) << "," << get<1>(p) << "," << get<2>(p) << "> ";
  }
  cout << "\n";
}

void print_multiset(const multiset<pair<int, int>> &m) {
  for (auto const &p : m) {
    cout << "<" << p.first << "," << p.second << "> ";
  }
  cout << "\n";
}

void mqsort(auto iit, auto oit) {
  if (iit == oit)
    return;
  auto pivot = *(oit - 1);
  // XXX: All less than pivot
  auto tt = partition(iit, oit, [&pivot](auto x) { return x < pivot; });
  // XXX: All >= pivot
  auto ss = partition(tt, oit, [&pivot](auto x) { return !(x < pivot); });
  mqsort(iit, tt); // For <
  mqsort(ss, oit); // For >=
}

int possible_sticks() {
  size_t N;
  cin >> N;

  vector<int> s(N, 0);

  for (size_t i = 0; i < N; ++i)
    cin >> s[i];

  int maxv = *max_element(s.begin(), s.end());

  // XXX: Optimisation instead of using 10^9
  int mf = -1;
  size_t ifreq = 0;
  for (int i = 1; i <= maxv; ++i) {
    size_t ifreqn =
        count_if(s.begin(), s.end(), [&i](int x) { return x == i; });
    // XXX: No two values with the same frequency
    assert(ifreq != ifreqn);
    if (ifreqn > ifreq) {
      mf = i;
      ifreq = ifreqn;
    }
  }

  int min_cost = 0;
  for (int x : s) {
    if (x != mf) {
      min_cost += abs(mf - x);
    }
  }

  cout << min_cost << "\n";
  return 0;
}

void decToBinary(int n, vector<int> &binaryNum) {
  // vector<int> binaryNum(n, 0);

  // Counter for binary array
  int i = 0;
  while (n > 0) {
    // Storing remainder in binary
    // array
    binaryNum[i] = n % 2;
    n = n / 2;
    i++;
  }
}

int gray_code() {
  size_t N;
  cin >> N;
  assert(N >= 1);
  size_t value = pow(2, N);
  vector<int> binaryNum(N, 0);
  for (size_t i = 0; i < value; ++i) {
    decToBinary(i, binaryNum);
    print_vec(binaryNum);
  }
  return 0;
}

int list_to_set() {
  size_t N;
  cin >> N;
  vector<int> nums(N, 0);
  for (size_t i = 0; i < N; ++i)
    cin >> nums[i];

  cout << unordered_set<int>(nums.begin(), nums.end()).size();

  cout << "\n";

  return 0;
}

void sum_of_two() {
  size_t N;

  int X; // can be negative

  cin >> N;
  cin >> X;

  vector<int> ws;
  int w;
  while (cin >> w) {
    ws.push_back(w);
  }

  size_t counter = 0;
  bool done = false;
  unordered_map<int, size_t> orig;
  // XXX: This is O(N)
  for (int w : ws) {
    int c = X - w;
    if (orig.contains(c)) {
      cout << orig[c] << " " << (counter + 1) << "\n";
      done = true;
      break;
    }
    orig[w] = ++counter;
  }

  if (!done)
    cout << "IMPOSSIBLE";
  cout << "\n";
}

void fwheel() {
  size_t N;
  int W;

  cin >> N;
  cin >> W;

  vector<int> ws;
  int w;
  while (cin >> w)
    ws.push_back(w);

  // XXX: The list of children weight is sorted now!
  sort(begin(ws), end(ws));

  size_t g = 0; // number of gondolas
  for (auto it = begin(ws); it != end(ws);) {
    // XXX: Iterate through the list of sorted weights
    for (auto rit = rbegin(ws); rit != rend(ws);) {
      if (*it + *rit <= W) {
        // XXX: Fit these two children in the gondola
        ++it;
        ++rit;
      } else
        ++rit; // fat kid in a single gondola.
      g += 1;
      // XXX: Break if rit and it are past each other or equal.
      size_t itpos = it - begin(ws);
      size_t ritpos = rend(ws) - rit;
      if (ritpos <= itpos)
        goto U;
    }
  }
U:
  cout << g << "\n";
}

// XXX: O(N-K+1)^2
int apart(auto f, auto l, size_t N, size_t K) {
  assert((l - f) >= K);

  if (K == 1)
    return accumulate(f, l, 0);

  int cmax = numeric_limits<int>::max();
  int acc = 0;
  // XXX: remaining items in the vector are (l - (f+1))
  while (l - (f + 1) >= K - 1) {
    acc += *f;
    int res = apart(f + 1, l, N - 1, K - 1);
    int t = max(acc, res);
    cmax = cmax > t ? t : cmax;
    f += 1;
  }
  return cmax;
}

void ad_sup() {

  size_t N; // size of array
  size_t K; // partitions

  cin >> N;
  cin >> K;

  vector<int> a(N, 0);

  int w;
  size_t counter = 0;
  while (cin >> w) {
    a[counter] = w;
    counter += 1;
  }

  // Now partition it.
  cout << apart(begin(a), end(a), N, K) << "\n";
}

void slide_median() {
  size_t N, K;
  cin >> N;
  cin >> K;

  vector<int> ints(N, 0);
  size_t counter = 0;
  int w;
  while (cin >> w)
    ints[counter++] = w;

  // XXX: Now just make a O((N-K+1)*(K*log K)) algorithm.
  size_t kd2 = K / 2;
  size_t kp2 = K % 2;
  vector<int> cp;
  for (size_t i = 0; i <= ints.size() - K; ++i) {
    // XXX: First copy the K elements into another vector
    // XXX: Reusing the space, so OK! space: O(K)
    cp = vector<int>(begin(ints) + i, begin(ints) + i + K);
    sort(begin(cp), end(cp));
    if ((kp2) != 0) {
      cout << cp[kd2] << " ";
    } else {
      cout << min(cp[kd2], cp[kd2 - 1]) << " ";
    }
  }
  cout << "\n";
}

void polygon_area() {

  size_t N;
  cin >> N;

  vector<pair<int, int>> vertices(N, pair<int, int>{0, 0});
  size_t counter = 0;
  int x, y;
  while (cin >> x) {
    cin >> y;
    vertices[counter++] = pair<int, int>{x, y};
  }

  // XXX: Now make triangles and apply heron' formula
  pair<int, int> p1 = vertices[0];
  auto length = [](const pair<int, int> &x, const pair<int, int> &y) {
    return sqrt((y.first - x.first) * (y.first - x.first) +
                (y.second - x.second) * (y.second - x.second));
  };
  auto heron = [](const auto &l1, const auto &l2, const auto &l3) {
    auto s = 0.5 * (l1 + l2 + l3); // semi-perimeter
    return sqrt(s * (s - l1) * (s - l2) * (s - l3));
  };
  auto area = 0;
  for (auto it = begin(vertices) + 1; end(vertices) - it >= 2; ++it) {
    // XXX: compute the lengths of the sides
    auto l1 = length(p1, *it);
    auto l2 = length(p1, *(it + 1));
    auto l3 = length(*it, *(it + 1));
    // XXX: Now apply heron' formula
    area += heron(l1, l2, l3);
  }

  cout << 2 * area << "\n";
}

void min_euclid_dist() {
  size_t N;
  cin >> N;

  vector<pair<int, int>> ps(N, {0, 0});
  size_t counter = 0;
  int x, y;
  while (cin >> x) {
    cin >> y;
    ps[counter++] = {x, y};
  }
  // print_vec(ps);

  // XXX: First sort by xs in pair projection.
  // O(N*logN)
  sort(begin(ps), end(ps),
       [](pair<int, int> x, pair<int, int> y) { return x.first <= y.first; });

  // XXX: Get the min x distance in the sorted list
  int mxd = numeric_limits<int>::max();
  size_t min_xindex;
  // XXX: O(N)
  for (counter = 0, min_xindex = 0; counter < ps.size() - 1; ++counter) {
    int temp = (ps[counter + 1].first - ps[counter].first);
    if (temp <= mxd) {
      min_xindex = counter;
      mxd = temp;
    }
  }
  // XXX: Now get the total distance between the (x, y)-points at min_xindex;
  int fdist = ((ps[min_xindex].first - ps[min_xindex + 1].first) *
                   (ps[min_xindex].first - ps[min_xindex + 1].first) +
               (ps[min_xindex].second - ps[min_xindex + 1].second) *
                   (ps[min_xindex].second - ps[min_xindex + 1].second));

  // XXX: Do the same for min y distance
  // O(N*logN)
  sort(begin(ps), end(ps),
       [](pair<int, int> x, pair<int, int> y) { return x.second <= y.second; });

  // XXX: Get the min x distance in the sorted list
  int myd = numeric_limits<int>::max();
  size_t min_yindex;
  // O(N)
  for (counter = 0, min_yindex = 0; counter < ps.size() - 1; ++counter) {
    int temp = (ps[counter + 1].first - ps[counter].first);
    if (temp <= myd) {
      min_yindex = counter;
      myd = temp;
    }
  }
  // XXX: Now get the total distance between the (x, y)-points at min_xindex;
  int sdist = ((ps[min_yindex].first - ps[min_yindex + 1].first) *
                   (ps[min_yindex].first - ps[min_yindex + 1].first) +
               (ps[min_yindex].second - ps[min_yindex + 1].second) *
                   (ps[min_yindex].second - ps[min_yindex + 1].second));

  // XXX: Total runtime: O(N*(log N + 1))
  cout << min(fdist, sdist) << "\n";
}

template <typename T> void print_array(const vector<T> *a, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    cout << i << ": ";
    print_vec(a[i]);
  }
}

void shortest_route_1() {
  size_t V, E;
  cin >> V;
  cin >> E;

  // XXX: Now the array as a adjacency list
  vector<pair<int, int>> al[V];
  int v1, v2, c;
  while (cin >> v1) {
    cin >> v2;
    cin >> c;
    al[--v1].push_back({--v2, c});
  }
  // XXX: There are parallel edges in the graph!
  // print_array(&al[0], V);

  vector<int> processed(V, 0);
  using dist = int;
  using index = int;
  multiset<pair<dist, index>> dists;

  // XXX: First step of Dijkstra' algo
  for (size_t i = 1; i < V; ++i) {
    // XXX: dist to all the nodes from the first node.
    auto it =
        find_if(al[0].begin(), al[0].end(),
                [&i](const pair<index, dist> &x) { return x.first == i; });
    if (it != al[0].end())
      dists.insert({it->second, it->first});
    else
      dists.insert({INFINITY, i});
  }

  // XXX: Now start doing dijkstra' algo
  while (!dists.empty()) {
    // XXX: Take the first value from the sorted values.
    pair<dist, index> top = *dists.begin();
    // XXX: Removed the top value;
    dists.erase(begin(dists));
    // XXX: Push the cost to top in processed
    processed[top.second] = top.first;
    // XXX: Now update the values of the adjacent nodes in dists.
    for (auto it = begin(al[top.second]); it != end(al[top.second]); ++it) {
      auto it2 = find_if(begin(dists), end(dists), [&it](pair<dist, index> x) {
        return it->first == x.second;
      });
      pair<dist, index> nv{min(it2->first, (top.first + it->second)),
                           it2->second};
      dists.erase(it2);
      dists.insert(nv);
    }
  }
  print_vec(processed);
}

template <typename T>
void dfs_revert(const vector<T> *adj, const T s, vector<bool> &vis,
                vector<T> *ret, vector<T> &order, const int parent) {
  if (vis[s] && find(begin(order), end(order), s) == end(order)) {
    // XXX: However if you are not on the order vector, then this is a
    // cycle and you should add
    if (parent != -1)
      ret[s].push_back(parent);
    else
      assert(false);
    return;
  }
  if (!vis[s]) {
    vis[s] = true;
    // XXX: Add the children of the node to ret
    for (size_t i = 0; i < adj[s].size(); ++i) {
      ret[adj[s][i]].push_back(s);
      dfs_revert(adj, adj[s][i], vis, ret, order, s);
    }
    order.push_back(s);
  }
}

// XXX: This does both: (1) toplogical sort and inverting the graph/tree
// (cycles are handled correctly).
template <typename T>
void invert_adj_list(vector<T> *adj, size_t M, vector<T> *ret,
                     vector<T> &order) {
  // XXX: Use DFS to reverse
  vector<bool> vis(M, false);
  int parent = -1;
  for (size_t i = 0; i < M; ++i) {
    dfs_revert<int>(adj, i, vis, ret, order, parent);
  }
}

void kdf1(const vector<int> *adj, vector<bool> &vis, vector<int> &order,
          const int s) {
  if (!vis[s]) {
    vis[s] = true;
    // XXX: For each child of s do a dfs
    for (int c : adj[s]) {
      kdf1(adj, vis, order, c);
    }
    // XXX: Add the node to the order list
    order.push_back(s);
  }
}

void giant_pizza() {
  size_t N, M;
  cin >> N; // constraints
  cin >> M; // choices

  // XXX: first plus choice then second choice
  char s1, s2;
  int v1, v2;
  size_t counter = 0;
  size_t IMPLG_SIZE = M * 2;
  vector<int> implg[IMPLG_SIZE];
  vector<int> rev_implg[IMPLG_SIZE];

  auto get_not = [&M](int x) {
    if (x >= M) {
      return x - M;
    } else {
      return x + M;
    }
  };
  while (counter < N) {
    cin >> s1;
    cin >> v1;
    --v1;
    cin >> s2;
    cin >> v2;
    --v2;
    if (s1 == '-') {
      v1 = M + v1;
    }
    if (s2 == '-') {
      v2 = M + v2;
    }
    implg[get_not(v1)].push_back(v2);
    implg[get_not(v2)].push_back(v1);
    ++counter;
  }
  // print_array(implg, IMPLG_SIZE);
  vector<int> order;
  invert_adj_list(implg, IMPLG_SIZE, rev_implg, order);
  // cout << order << "\n";
  // print_array(rev_implg, IMPLG_SIZE);

  // XXX: Now get the sccs going in top-order
  vector<vector<int>> scc;
  vector<bool> vis(IMPLG_SIZE, false);
  for (int i = order.size() - 1; i >= 0; --i) {
    vector<int> sc;
    kdf1(rev_implg, vis, sc, order[i]);
    if (!sc.empty())
      scc.push_back(sc);
  }
  // cout << scc << scc.size() << "\n";
  // XXX: If the size of scc is the same as IMPLG_SIZE, then there is no
  // scc and hence, we can just go in topological order and give the
  // booleans a value.
  vector<char> result(M, ' ');
  vector<bool> filled(M, false);
  if (scc.size() == IMPLG_SIZE) {
    for (auto &sc : scc) {
      if (all_of(begin(filled), end(filled), [](bool value) { return value; }))
        break;
      assert(sc.size() == 1);
      if (sc[0] > M) {
        result[sc[0] - M] = '-';
      } else {
        result[sc[0]] = '+';
      }
      filled[sc[0]] = true;
    }
  } else {
    // XXX: We need to check if some a and !a occur in a scc, for all a.
    // FIXME: The above statement can be done using simple DFS.
  }
  cout << result << "\n";
}

// XXX: This is still O(V^2), because of find being O(V)
void DFS(const vector<int> *adj, int i, vector<int> &res, vector<bool> &vis) {
  // XXX: Base case
  if (vis[i]) {
    // XXX: Check if the visited node is already on the returning
    // result.
    if (find(begin(res), end(res), i) == end(res))
      cout << "IMPOSSIBLE\n";
    return;
  }
  vis[i] = true;
  // XXX: Now go through all the children and call DFS on them
  for (int v : adj[i]) {
    DFS(adj, v, res, vis);
  }
  // XXX: Now add yourself onto the res.
  res.push_back(i);
}

// XXX: Make a O(|V|+|E|) algo for topological sort with cycle
// detection.
void top_sort(const vector<int> *adj, size_t M, vector<int> &res) {
  vector<bool> vis(M, false);
  res.clear();
  for (size_t i = 0; i < M; ++i) {
    DFS(adj, i, res, vis);
  }
  reverse(begin(res), end(res));
}

void course_schedule() {
  size_t M, N;
  cin >> M; // number of courses.
  cin >> N; // number of constraints.

  size_t counter = 0;
  int n, m;

  // XXX: Adjacency list.
  vector<int> adj[M];

  while (counter < N) {
    cin >> n;
    n--;
    cin >> m;
    m--;
    adj[n].push_back(m);
    ++counter;
  }

  // XXX: Now just do a topological sort.
  vector<int> result(M, -1);

  top_sort(adj, M, result);

  // XXX: Add 1 to everything in the result
  transform(begin(result), end(result), begin(result),
            [](int x) { return x + 1; });
  print_vec(result);
}

void road_reparation() {
  // XXX: This is basically building a minimum spanning tree.
  // XXX: Use Kruskal's algorithm.

  size_t N, M;
  cin >> N >> M;

  size_t counter = 0;
  using weight = int;
  using vertex = int;
  using tt = tuple<weight, vertex, vertex>;
  priority_queue<tt, vector<tt>, greater<tt>> edges;
  weight w;
  vertex v1, v2;

  vector<vertex> adj[N];

  while (counter < M) {
    cin >> v1;
    --v1;
    cin >> v2;
    --v2;
    cin >> w;
    edges.push({w, v1, v2});
    ++counter;
  }

  // XXX: Now just kruskal' algo
  int ans = 0;
  // XXX: Complexity os O(|E|*log E)
  while (!edges.empty()) {
    tt top = edges.top();
    vector<vertex> *f = &adj[get<1>(top)];
    vector<vertex> *s = &adj[get<2>(top)];
    if (f->size() == 0 || s->size() == 0) {
      // cout << "allocating to adj list\n";
      f->push_back(get<2>(top));
      s->push_back(get<1>(top));
      ans += get<0>(top);
    }
    edges.pop();
  }
  cout << ans << "\n";
}

template <typename T>
auto discrete_binary_search(const auto sit, const auto eit, const T v) {
  // XXX: This only works for discrete types!
  auto toret = eit;
  auto it = lower_bound(sit, eit, v);
  // XXX: Only if you found the first occurance of value of type v
  // and it is exactly equal to the value given
  if (it != eit && *it == v)
    toret = it;
  return toret;
}

static inline int contains(const vector<vector<int>> &v, const int s,
                           const int i) {
  int toret = v.size();
  for (size_t j = i; j < v.size(); ++j) {
    auto it = discrete_binary_search(begin(v[j]), end(v[j]), s);
    if (it != end(v[j])) {
      toret = j;
      break;
    }
  }
  return toret;
}

void road_construction() {
  size_t M, N;
  cin >> N >> M;
  // XXX: disjoint union on sets
  vector<vector<int>> ds;
  // XXX: to start with every set has a single element.
  int counter = 1;
  for (size_t j = 0; j < N; ++j) {
    ds.push_back({counter});
    counter++;
  }

  // XXX: Now read the edges one after another.
  counter = 0;
  int ef, el;
  vector<pair<int, int>> edges;
  while (counter < M) {
    cin >> ef >> el;
    edges.push_back({min(ef, el), max(ef, el)});
    ++counter;
  }
  counter = 0;
  // XXX: O(M*(2 * N * log N))
  while (counter < M) {
    // XXX: O(N*log(N))
    auto [f, s] = edges[counter];
    // XXX: O(N)
    for (size_t i = 0; i < ds.size(); ++i) {
      // XXX: First check if both f and s are in the same set, then do
      // nothing.
      auto it = discrete_binary_search(begin(ds[i]), end(ds[i]), f);
      if (it != end(ds[i])) {
        auto it = discrete_binary_search(begin(ds[i]), end(ds[i]), s);
        if (it != end(ds[i])) {
          break;
        } else {
          // XXX: Get the terminating vertex in the rest of the vectors!
          auto index = contains(ds, s, (i + 1));
          for (size_t k = 0; k < ds[index].size(); ++k) {
            ds[i].push_back(ds[index][k]);
          }
          sort(begin(ds[i]), end(ds[i]));
          // XXX: Delete the vector at position index
          ds.erase(begin(ds) + index);
        }
        break;
      }
    }
    vector<int> sizes;
    transform(begin(ds), end(ds), back_inserter(sizes),
              [](const vector<int> &vv) { return vv.size(); });
    cout << ds.size() << " " << *max_element(begin(sizes), end(sizes)) << "\n";
    ++counter;
  }
}

template <typename T>
bool reachable(const vector<T> *adj, const T s, const T d, vector<bool> &vis) {
  bool toret = false;
  if (s == d) {
    toret = true;
  }
  // XXX: Check if the destination is reachable from the source using
  // DFS on adj.
  else if (!vis[s]) {
    vis[s] = true;
    for (size_t i = 0; i < adj[s].size(); ++i) {
      toret = reachable(adj, adj[s][i], d, vis);
      if (toret)
        break;
    }
  }
  return toret;
}

void flight_route_reqs() {
  size_t N, M;
  cin >> N;
  cin >> M;

  vector<int> adj[N];

  size_t counter = 0;
  int s, d;
  vector<pair<int, int>> edges;
  while (counter < M) {
    cin >> s;
    --s;
    cin >> d;
    --d;
    edges.push_back({s, d});
    ++counter;
  }
  counter = 0;
  size_t min_edges = 0;
  for (auto &[s, d] : edges) {
    vector<bool> vis(N, false);
    if (!reachable<int>(adj, s, d, vis)) {
      adj[s].push_back(d);
      min_edges++;
    }
  }
  cout << min_edges << "\n";
}

void forbidden_cities() {
  size_t C, R, Q;
  cin >> C >> R >> Q;

  size_t counter = 0;
  vector<pair<int, int>> edges;
  int s, d;
  while (counter < R) {
    cin >> s;
    s--;
    cin >> d;
    d--;
    edges.push_back({s, d});
    ++counter;
  }

  counter = 0;
  // XXX: Get all the queries
  vector<tuple<int, int, int>> queries;
  int f;
  while (counter < Q) {
    cin >> s;
    --s;
    cin >> d;
    --d;
    cin >> f;
    --f;
    queries.push_back({s, d, f});
    ++counter;
  }
  // XXX: Now build a graph for each query separately and then find
  // reachbility.
  for (const tuple<int, int, int> &q : queries) {
    vector<int> adj[C];
    s = get<0>(q);
    d = get<1>(q);
    f = get<2>(q);
    // XXX: Add the edges to the adjacency list, iff either of the edges
    // are not in the forbidden list and then do reachbilty.
    for (auto &[rs, rd] : edges) {
      if (rs != f && rd != f) {
        adj[rs].push_back(rd);
        adj[rd].push_back(rs);
      }
    }
    // XXX: Now do reachbility
    vector<bool> vis(C, false);
    if (reachable<int>(adj, s, d, vis))
      cout << "YES\n";
    else
      cout << "NO\n";
  }
}

void new_flight_routes() {
  size_t N, M;
  cin >> N; // cities
  cin >> M; // directed routes

  vector<int> adj1[N];
  vector<int> adj2[N];
  size_t counter = 0;
  int s, d;
  while (counter < M) {
    cin >> s;
    --s;
    cin >> d;
    --d;
    // XXX: This is the main graph
    adj1[s].push_back(d);
    // XXX: This is the inverted graph
    adj2[d].push_back(s);
    ++counter;
  }
  // XXX: Now make a topological sorted list
  vector<bool> vis(N, false);
  vector<int> order;
  for (int i = 0; i < N; ++i) {
    kdf1(adj1, vis, order, i);
  }
  // XXX: Now move in order in adj2 and get the strongly connected
  // components.
  vector<vector<int>> scc;
  fill(begin(vis), end(vis), false);
  for (int i = order.size() - 1; i >= 0; --i) {
    vector<int> sc;
    kdf1(adj2, vis, sc, order[i]);
    if (!sc.empty())
      scc.push_back({sc});
  }
  // XXX: Now we just make a route from last one in scc to last one in
  // the one before!
  int nroutes = scc.size() - 1;
  cout << nroutes << "\n";
  for (int i = scc.size() - 1; i >= 1; --i) {
    cout << scc[i][scc[i].size() - 1] + 1 << " "
         << scc[i - 1][scc[i - 1].size() - 1] + 1 << "\n";
  }
}

// XXX: This is a dynamic programming solution to grid search!
// XXX: It does memoization with DFS to destination
pair<bool, bool> search_grid(int u, int dest, vector<int> &path,
                             vector<bool> &vis, const vector<int> *adj,
                             vector<vector<int>> *memo, vector<int> *ndone,
                             const int start, const int tnodes) {
  // cout << "start: " << start << " dest: " << dest << "\n";
  // cout << "path: " << path;
  // cout << "u: " << u << "\n";
  // cout << "vis: " << vis;
  bool done = false;
  bool found = false;
  // XXX: Base case
  if (u == dest) {
    if (memo[u].empty())
      memo[u].push_back({u});
    found = true;
    // XXX: This is a short cut while traversing!
    // if (all_of(begin(vis), end(vis), [](bool x) { return x; })) {
    //   done = true;
    // }
  } else {
    for (st i = 0; i < adj[u].size(); ++i) {
      auto it = find(begin(ndone[u]), end(ndone[u]), adj[u][i]);
      // cout << "Checking child: " << adj[u][i] << "\n";
      if (!vis[adj[u][i]] && (it == end(ndone[u]))) {
        // cout << "Going to child: " << adj[u][i] << "\n";
        path.push_back(adj[u][i]);
        vis[adj[u][i]] = true;
        // XXX: Also state that you have done this child
        ndone[u].push_back(adj[u][i]);
        auto [dd, ff] = search_grid(adj[u][i], dest, path, vis, adj, memo,
                                    ndone, start, tnodes);
        if (dd) {
          // cout << "done!\n";
          done = true;
          goto END;
        } else if (ff) {
          // cout << "ff is true\n";
          // XXX: I have found another path to the destination
          // XXX: Go through all child paths and add them to yourself
          // cout << path << "\n";
          // cout << "Checking to put in: " << memo[adj[u][i]].size() << " in "
          //      << u << "\n";
          // XXX: Get only the longest vector from the child!
          vector<int> sizes(memo[adj[u][i]].size(), 0);
          transform(begin(memo[adj[u][i]]), end(memo[adj[u][i]]), begin(sizes),
                    [](const vector<int> &vvv) { return vvv.size(); });
          // XXX: Now get the max size
          int max_child_size = *max_element(begin(sizes), end(sizes));
          for (const vector<int> &v : memo[adj[u][i]]) {
            // XXX: This path does not already exist in this node
            // XXX: Make sure that the vector does not contain this node!
            // XXX: The current node is prepended to the path
            if (v.size() == max_child_size) { // just using the longest one(s)
              if (find(begin(v), end(v), u) == end(v)) {
                vector<int> vv(v.size() + 1, u);
                copy(begin(v), end(v), begin(vv) + 1);
                memo[u].push_back(vv);
              }
            }
          }
          if (u == start) {
            for (const auto &v : memo[u]) {
              if (v.size() == tnodes) {
                found = true;
                goto END;
              }
            }
          }
          found = true;
        }
      }
    }
  }
  // XXX: If we have some path to the destination
  // cout << "memo: ";
  // for (const vector<int> &v : memo[u]) {
  //   cout << v;
  // }
  if (!memo[u].empty())
    found = true;
  // XXX: Move yourself out of the path and set vis to false
  path.erase(end(path) - 1);
  vis[u] = false;
END:
  // cout << "returning from : " << u << " done: " << done << "\n";
  // cout << "returning from : " << u << " found: " << found << "\n";
  return {done, found};
}

void print_path(const vector<int> &path, int cols) {
  string spath = "";
  for (st i = 0; i < path.size() - 1; ++i) {
    int f = path[i];
    int s = path[i + 1];
    if (s - f == 1)
      spath += "R";
    else if (s - f == -1)
      spath += "L";
    else if (s - f == cols)
      spath += "D";
    else if (s - f == -cols)
      spath += "U";
    else
      throw "path wrong!";
  }
  cout << "YES\n";
  cout << spath << "\n";
}

// XXX: Done! Dynamic programming based search of the grid!
void grid_search() {
  st T;
  cin >> T;

  st counter = 0;
  tuple<int, int, int, int, int, int> Q[T];
  int r, c, x1, y1, x2, y2;
  while (counter < T) {
    cin >> r;
    cin >> c;
    cin >> x1;
    --x1;
    cin >> y1;
    --y1;
    cin >> x2;
    --x2;
    cin >> y2;
    --y2;
    Q[counter] = {r, c, x1, y1, x2, y2};
    ++counter;
  }
  // XXX: Now build the adj list for each query separately.
  counter = 0;
  while (counter < T) {
    st rows = get<0>(Q[counter]);
    st cols = get<1>(Q[counter]);
    vector<int> adj[rows * cols];
    st nc = 0;
    // XXX: Make the adj list
    for (st i = 0; i < rows; ++i) {
      for (st j = 0; j < cols; ++j) {
        nc = (i * cols) + j;
        if (j != 0) {
          // XXX: Make the left node
          adj[nc].push_back(nc - 1);
        }
        if (j != (cols - 1)) {
          // XXX: Make the right node
          adj[nc].push_back(nc + 1);
        }
        if (i != 0) {
          // XXX: Make the up node
          adj[nc].push_back(nc - cols);
        }
        if (i != (rows - 1)) {
          // XXX: Make the down node
          adj[nc].push_back(nc + cols);
        }
      }
    }
    // print_array(adj, cols * rows);
    // XXX: Now we can perform the dfs search to get the path!
    st start = (get<2>(Q[counter]) * cols + get<3>(Q[counter]));
    st end = (get<4>(Q[counter]) * cols + get<5>(Q[counter]));
    // cout << "s: " << start << " e: " << end << "\n";
    vector<int> path;
    vector<bool> vis(rows * cols, false);
    // XXX: Add the start to the path and set it's vis to true
    path.push_back(start);
    vis[start] = true;
    vector<vector<int>> memo[rows * cols];
    vector<int> ndone[rows * cols];
    auto [ret, found] = search_grid(start, end, path, vis, adj, memo, ndone,
                                    start, rows * cols);
    // cout << "ret: " << ret << " found: " << found << "\n";
    if (ret) {
      print_path(path, cols);
    } else if (found) {
      // XXX: Check if there is any path to destination that includes all the
      // nodes (rows * cols)
      for (const vector<int> &c : memo[start]) {
        if (c.size() == (rows * cols)) {
          ret = true;
          print_path(c, cols);
          break;
        }
      }
    }
    if (!ret)
      cout << "NO\n";
    ++counter;
  }
}

int bfs_max_flow(int s, const int N, const vector<int> *adj, const int *cap,
                 vector<int> &parent, const int t) {
  int flow = 0;
  // XXX: initialise the parent vector
  fill(begin(parent), end(parent), -1);
  // XXX: Set the source parent to -2, something negative.
  parent[s] = -2;
  // XXX: Make the queue to perform BFS
  queue<pair<int, int>> q;
  q.push({s, INFINITY});
  while (!q.empty()) {
    // XXX: Start processing the nodes
    // XXX: Need to get the child with cap > 0 if it exists.
    auto [p, pflow] = q.front();
    q.pop();
    for (const int &c : adj[p]) {
      if (parent[c] == -1 and cap[p * N + c] > 0) {
        pflow = min(pflow, cap[p * N + c]);
        // cout << "pflow: " << pflow << "\n";
        parent[c] = p; // set the parent of the visited node
        if (c == t) {
          // XXX: Found the target node!
          flow = pflow;
          // cout << "flow: " << flow << "\n";
          goto END;
        } else {
          // XXX: Push it onto the q to process.
          q.push({c, pflow});
        }
      }
    }
  }
END:
  return flow;
}

void get_reachable_edges_cap_gt0(const int s, const vector<int> *adj,
                                 const int *cap, vector<int> &r, const int N,
                                 vector<bool> &vis) {
  vis[s] = true;
  // XXX: Go through each child
  for (const int &n : adj[s]) {
    if (!vis[n]) {
      // XXX: Now n has to be the child
      if (cap[s * N + n] > 0) {
        get_reachable_edges_cap_gt0(n, adj, cap, r, N, vis);
      }
    }
  }
  r.push_back(s);
}

void stcut(const vector<int> *adj, const int N, const int *cap,
           vector<pair<int, int>> &edges, const vector<int> &sr) {

  for (const int &n : sr) {
    // XXX: Get edges that have negative capacity
    for (const int &c : adj[n]) {
      if ((find(begin(sr), end(sr), c) == end(sr)) and cap[n * N + c] <= 0) {
        edges.push_back({n, c});
      }
    }
  }
}

// XXX: This is max-flow algorithm. Think of it as how much max water
// can flow from the source tap (s) to the destination tank (t). Such
// that none of the pipes burst! The s-t cut, is like cutting the pipes,
// which highlight the bottleneck!
void police_chase() {
  st N, M;
  cin >> N; // nodes in graph
  cin >> M; // edges in graph

  int cap[N][N]; // the capacity matrix of each edge.
  fill(&cap[0][0], &cap[0][0] + (N * N), -1);
  vector<int> adj[N];
  st counter = 0;
  int s = 0;
  int t = N - 1;
  int r, d;
  while (counter < M) {
    cin >> r;
    --r;
    cin >> d;
    --d;
    adj[r].push_back(d);
    adj[d].push_back(r);
    // XXX: Fill in the capacity for one side
    cap[r][d] = 1;
    cap[d][r] = 0;
    ++counter;
  }

  // print_iter(&adj[0], &adj[0] + N);
  // for (st i = 0; i < N; ++i) {
  //   for (st j = 0; j < N; ++j) {
  //     cout << i << "," << j << ": " << cap[i][j] << " ";
  //   }
  //   cout << "\n";
  // }

  // XXX: Now start doing the max-flow
  // int max_flow = 0;
  vector<int> parent(N, -1);
  while (1) {
    int flow = bfs_max_flow(s, N, adj, &cap[0][0], parent,
                            t); // obtained from bfs_max_flow
    // cout << "parent: " << parent;
    // cout << "flow: " << flow << "\n";
    if (flow == 0)
      break; // done with max_flow
    // XXX: Update the capacities
    // max_flow += flow;
    // XXX: start from the target node and move backwards (like Dijkstra)
    int c = t;
    while (c != s) {
      int p = parent[c];
      cap[p][c] -= flow; // capacity is reduced by flow.
      cap[c][p] += flow; // cap in opposite direction is now positive.
      c = p;
    }
    // for (st i = 0; i < N; ++i) {
    //   for (st j = 0; j < N; ++j) {
    // 	cout << i << "," << j << ": " << cap[i][j] << " ";
    //   }
    //   cout << "\n";
    // }
  }
  // cout << max_flow;

  // XXX: Now we have the max_flow. Now we want to get the s-t cut. This
  // is done by doing a dfs from source, such that capacity of the edges
  // from source > 0.
  vector<int> sr;
  vector<bool> vis(N, false);
  get_reachable_edges_cap_gt0(s, adj, &cap[0][0], sr, N, vis);
  // XXX: Now we have the edges reachable from source
  // XXX: Now cut the edges from sr
  vector<pair<int, int>> edges;
  stcut(adj, N, &cap[0][0], edges, sr);
  cout << edges.size() << "\n";
  for (auto &[f, s] : edges) {
    cout << (f + 1) << " " << (s + 1) << "\n";
  }
}

// XXX: This is the bipartite graph matching algorithm. Basic idea is to
// match one node from left set to the one to the right set, such that
// we have maximum number of matches.
void school_dance(){
  st B, G, P;
  cin >> B; // number of boys
  cin >> G; // number of girls
  cin >> P; // potential pairs

  // XXX: We have boys in 1 set (left). Girls in 1 set (right).
  // Potential matchings. We have to match 1 boy/girl so that there are
  // no conflicts and maximum matches are found.

  // XXX: This will be solved using max_flow Ford_Fulkerson algorithm.

  st counter = 0;
  int s = B + G; // the source node.
  int t = s + 1; // the target node.
  int N = B + G + 2;  // total number of nodes in the graph.
  vector<int> adj[N]; // graph adjacency list format.
  int cap[N][N];      // the capacity of edges graph O(N * N)
  fill(&cap[0][0], &cap[0][0] + (N * N), -1);
  int b, g;

  // TODO: I don't think I need to have an undirected graph!
  while (counter < P) {
    cin >> b;
    --b;
    cin >> g;
    --g;
    g += B;
    // XXX: source node connected to each boy (left)
    if(find(begin(adj[s]), end(adj[s]), b) == end(adj[s]))
      adj[s].push_back(b);
    if (find(begin(adj[b]), end(adj[b]), s) == end(adj[b]))
      adj[b].push_back(s);
    // XXX: Fill in the capacity matrix
    cap[s][b] = 1;
    cap[b][s] = 0;

    // XXX: Girl (right set) connected to the target
    if (find(begin(adj[g]), end(adj[g]), t) == end(adj[g]))
      adj[g].push_back(t);
    if (find(begin(adj[t]), end(adj[t]), g) == end(adj[t]))
      adj[t].push_back(g);

    // XXX: Fill in the capacity matrix
    cap[g][t] = 1;
    cap[t][g] = 0;

    // XXX: Boy connected to the girl
    adj[b].push_back(g);
    adj[g].push_back(b);
    // XXX: Fill in the capacity matrix
    cap[b][g] = 1;
    cap[g][b] = 0;

    ++counter;
  }

  print_iter(&adj[0], &adj[0] + N);
  for (st i = 0; i < N; ++i) {
    for (st j = 0; j < N; ++j) {
      cout << i << "," << j << ": " << cap[i][j] << " ";
    }
    cout << "\n";
  }

  // XXX: Now do bfs_max_flow
  int max_flow = 0;
  int flow = 0;
  vector<int> parent(N, -1);
  while (1) {
    flow = bfs_max_flow(s, N, adj, &cap[0][0], parent, t);
    if (!flow)
      break;
    max_flow += flow;
    // XXX: Update the capacity matrix
    int c = t;
    cout << "parent: " << parent << "\n";
    while (c != s) {
      int p = parent[c];
      cap[p][c] -= flow;
      cap[c][p] += flow;
      c = p;
    }
  }
  cout << max_flow << "\n";
  // print_iter(&adj[0], &adj[0] + N);
  // for (st i = 0; i < N; ++i) {
  //   for (st j = 0; j < N; ++j) {
  //     cout << i << "," << j << ": " << cap[i][j] << " ";
  //   }
  //   cout << "\n";
  // }
  // XXX: Matches are those, that have a capacity of zero in the cap
  // matrix.

  for (st b = 0; b < B; ++b)
    for (st g = B; g < (B + G); ++g)
      if (!cap[b][g])
        cout << (b + 1) << " " << ((g -B) + 1) << "\n";
}

void string_reverse(string &s) {
  string toret = "";
  st j = s.size() - 1;
  for (st i = 0; i < s.size(); ++i, --j) {
    if (i >= j)
      break;
    char temp = s[i];
    s[i] = s[j];
    s[j] = temp;
  }
}

int main() {
  // list_to_set();
  // gray_code();
  // sum_of_two();
  // fwheel();
  // ad_sup();
  // slide_median();
  // polygon_area();
  // min_euclid_dist();
  // shortest_route_1();
  // course_schedule();
  // road_reparation();
  // road_construction();
  // flight_route_reqs();
  // forbidden_cities();
  // new_flight_routes();
  // giant_pizza();
  // grid_search();
  // police_chase();
  // school_dance();
  return 0;
}
