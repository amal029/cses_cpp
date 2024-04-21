#include <algorithm>
#include <assert.h>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <math.h>
#include <numeric>
#include <optional>
#include <ostream>
#include <queue>
#include <ranges>
#include <set>
#include <stack>
#include <string>
#include <sys/types.h>
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
enable_if<is_integral_v<U>, T>::type binary_find(T s, T e, const U v) {
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

// XXX: Generic tuple printing (from C++17 only)
template <typename... Ts>
std::ostream &operator<<(std::ostream &os, std::tuple<Ts...> const &theTuple) {
  std::apply(
      [&os](Ts const &...tupleArgs) {
        os << '[';
        std::size_t n{0};
        ((os << tupleArgs << (++n != sizeof...(Ts) ? ", " : "")), ...);
        os << ']';
      },
      theTuple);
  return os;
}

template <typename T, typename U>
ostream &operator<<(ostream &os, const pair<T, U> &vec) {
  os << "<" << vec.first << "," << vec.second << ">\n";
  return os;
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
  auto det = [](pair<int, int> f, pair<int, int> s) {
    auto &[x1, y1] = f;
    auto &[x2, y2] = s;
    return (x1 * y2) - (x2 * y1);
  };

  // XXX: Now make triangles and apply shoelace formula
  auto area = 0;
  for (auto it = begin(vertices) + 1; it != end(vertices); ++it) {
    // XXX: Apply shoelace formula for any polygon
    area += det(*(it - 1), *it);
  }
  area += det(*(end(vertices) - 1), *begin(vertices));

  // XXX: We are not dividing it by 2 as asked for!
  cout << area << "\n";
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
void school_dance() {
  st B, G, P;
  cin >> B; // number of boys
  cin >> G; // number of girls
  cin >> P; // potential pairs

  // XXX: We have boys in 1 set (left). Girls in 1 set (right).
  // Potential matchings. We have to match 1 boy/girl so that there are
  // no conflicts and maximum matches are found.

  // XXX: This will be solved using max_flow Ford_Fulkerson algorithm.

  st counter = 0;
  int s = B + G;      // the source node.
  int t = s + 1;      // the target node.
  int N = B + G + 2;  // total number of nodes in the graph.
  vector<int> adj[N]; // graph adjacency list format.
  int cap[N][N];      // the capacity of edges graph O(N * N)
  fill(&cap[0][0], &cap[0][0] + (N * N), -1);
  int b, g;

  while (counter < P) {
    cin >> b;
    --b;
    cin >> g;
    --g;
    g += B;
    // XXX: source node connected to each boy (left)
    if (find(begin(adj[s]), end(adj[s]), b) == end(adj[s]))
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
        cout << (b + 1) << " " << ((g - B) + 1) << "\n";
}

constexpr int cfib(int n) {
  if (n == 0)
    return 0;
  else if (n == 1)
    return 1;
  else
    return cfib(n - 1) + cfib(n - 2);
}

void teleporters_path() {
  st N, M;
  cin >> N; // nodes
  cin >> M; // edges
  st counter = 0;
  int s, t;
  st degree[N];
  fill(&degree[0], &degree[0] + N, 0);
  vector<pair<int, bool>> adj[N];
  while (counter < M) {
    cin >> s;
    --s;
    cin >> t;
    --t;
    // XXX: if s or d has a number increase the degree for that node
    degree[s] += 1;
    degree[t] += 1;
    // XXX: Add the adjacency list graph
    adj[s].push_back({t, false});
    ++counter;
  }
  // print_iter(&degree[0], &degree[0] + N);
  // print_iter(&adj[0], &adj[0] + N);

  // XXX: Now for a eulerain path either all nodes should have an even
  // degree or exactly 2 nodes with odd degree
  st ec = 0, oc = 0;
  for (const st &i : degree) {
    if (i % 2 == 0)
      ec += 1;
    else
      oc += 1;
  }
  if (ec != N) {
    if (oc != 2) {
      cout << "IMPOSSIBLE\n";
      return;
    }
  }
  // XXX: Now just find the Eulerian path
  int start = 0;
  stack<int> S;
  vector<int> path;
  S.push(start);
  while (!S.empty()) {
    // print_iter(&degree[0], &degree[0] + N);
    // XXX: Traverse the adjacency list of the node, if the edge is not deleted.
    int curr = S.top();
    // cout << "curr: " << curr << "\n";
    // cout << adj[curr];
    if (!degree[curr]) {
      // XXX: remove the node from the stack and put it on the path
      path.push_back(curr + 1);
      S.pop();
    } else {
      // XXX: Reduce the degree of curr
      degree[curr] -= 1;
      // XXX: Go through all the children and traverse them!
      for (auto &[c, del] : adj[curr]) {
        if (!del) {
          del = true;
          degree[c] -= 1;
          S.push(c);
          // XXX: Break, because we want to want to traverse one edge at a time!
          break;
        }
      }
    }
  }
  reverse(begin(path), end(path));
  cout << path;
}

// XXX: Bipartite graph division (only even cycles)
void building_teams() {
  st N, M;
  cin >> N;
  cin >> M;
  st counter = 0;
  vector<int> adj[N];
  int s, t;
  while (counter < M) {
    cin >> s;
    --s;
    cin >> t;
    --t;
    ++counter;
    // XXX: Undirected graph
    adj[s].push_back(t);
    adj[t].push_back(s);
  }
  // print_iter(adj, adj + N);
  // XXX: Make a recursive lambda (DFS)
  vector<bool> vis(N, false);
  vector<uint16_t> color(N, 3);
  // function<int(st counter)> dfs;
  auto dfs = [&adj, &vis, &color](auto &&self, int counter) -> int {
    vis[counter] = true;
    int8_t c = -1;
    bool toret = 0;
    // XXX: Check if all visited neighbors are true?
    for (int &x : adj[counter]) {
      if (vis[x]) {
        if (c == -1)
          c = color[x];
        else if (c != color[x]) {
          toret = 1;
          goto END;
        }
      }
    }
    // XXX: Assign yourself a color different to c
    if (c == -1)
      // XXX: Random choice
      color[counter] = 1;
    else if (c == 1)
      color[counter] = 2;
    else if (c == 2)
      color[counter] = 1;

    // XXX: Now go to the neighbor that is not visited
    for (int &x : adj[counter]) {
      if (!vis[x])
        toret = self(self, x);
      if (toret)
        break;
    }
  END:
    return toret;
  };
  // XXX: DO dfs for all nodes
  int ret = 0;
  for (st i = 0; i < N; ++i) {
    if (!vis[i])
      ret = dfs(dfs, i);
    if (ret)
      break;
  }
  if (ret)
    cout << "IMPOSSIBLE\n";
  else
    cout << color;
}

void convex_hull() {
  st N;
  cin >> N;
  st counter = 0;
  int x, y;
  typedef pair<int, int> point;
  vector<point> points;
  cin >> x;
  --x;
  cin >> y;
  --y;
  points.push_back({x, y});
  // XXX: Also get the min y point
  point min_point{x, y};
  // XXX: The first point is the min point
  while (counter < N - 1) {
    cin >> x;
    --x;
    cin >> y;
    --y;
    // XXX: If the curr value of y is less than min_point then update min_point
    if (y < min_point.second) {
      min_point = {x, y};
    } else if (y == min_point.second) {
      // XXX: Check the smaller x coordinate
      if (x <= min_point.first) {
        min_point = {x, y};
      }
    }
    points.push_back({x, y});
    ++counter;
  }
  // XXX: Sort the points according to their polar angle

  // XXX: First get the polar angle of each point
  auto dist = [](point x, point y) {
    auto d1 = (x.second - y.second) * (x.second - y.second);
    auto d2 = (x.first - y.first) * (x.first - y.first);
    return sqrt(d1 + d2);
  };
  auto angle = [&min_point](point p) {
    auto n = (p.second - min_point.second);
    auto d = (p.first - min_point.first);
    if (!d)
      return numeric_limits<int>::max();
    else
      return (n / d);
  };
  auto sort_by_angle_and_dist = [&dist, &min_point, &angle](point x, point y) {
    bool toret = false;
    auto a1 = angle(x);
    auto a2 = angle(y);
    if (a1 > a2)
      toret = true;
    else if (a1 == a2) {
      // auto d1 = dist(x);
      // auto d2 = dist(y);
      // toret = d1 <= d2;
    }
    return toret;
  };
  // XXX: Sorted the points in the hull!
  sort(begin(points), end(points), sort_by_angle_and_dist);
}

void point_line_location() {
  st T;
  cin >> T;
  st counter = 0;
  double x1, y1, x2, y2, x3, y3;
  while (counter < T) {
    cin >> x1;
    cin >> y1;
    cin >> x2;
    cin >> y2;
    cin >> x3;
    cin >> y3;

    // XXX: First get the slope
    double m = (y2 - y1) / (x2 - x1);
    // cout << "m: " << m << "\n";
    // XXX: Get the constant
    double c = y1 - (m * x1);
    // cout << "c: " << c << "\n";
    // Now get the location for a given x
    double _y = (m * x3) + c;
    if (_y == y3) {
      cout << "TOUCH\n";
    } else {
      // If slope is positive
      if (m > 0) {
        if (y3 < _y) {
          cout << "RIGHT\n";
        } else
          cout << "LEFT\n";
      } else if (m < 0) {
        // XXX: Slope is negative
        if (y3 < _y)
          cout << "LEFT\n";
        else
          cout << "RIGHT\n";
      }
    }
    ++counter;
  }
}

void line_segment_intersection() {
  st T;
  cin >> T;
  st counter = 0;
  double x1, y1, x2, y2, x3, y3, x4, y4;
  while (counter < T) {
    cin >> x1;
    cin >> y1;
    cin >> x2;
    cin >> y2;
    cin >> x3;
    cin >> y3;
    cin >> x4;
    cin >> y4;
    // XXX: Now compute the intersection
    double m1 = (y2 - y1) / (x2 - x1);
    double m2 = (y3 - y4) / (x3 - x4);
    // XXX: Get the constants
    double c1 = y1 - (m1 * x1);
    double c2 = y3 - (m2 * x3);

    // XXX: The the common point if any that leads to intersection of 2
    // lines!
    if ((m1 - m2) == 0) {
      if (c1 != c2)
        cout << "NO\n";
      else
        cout << "YES\n";
    } else {
      double x = (c2 - c1) / (m1 - m2);
      double y = (m1 * x) + c1;

      // cout << "x, y: " << x << "," << y << "\n";

      double minx = min(x1, x2);
      double maxx = max(x1, x2);
      double miny = min(y1, y2);
      double maxy = max(y1, y2);

      if (not(x >= minx and x <= maxx)) {
        cout << "NO\n";
        goto END;
      }
      if (not(y >= miny and y <= maxy)) {
        cout << "NO\n";
        goto END;
      }
      minx = min(x3, x4);
      maxx = max(x3, x4);
      miny = min(y3, y4);
      maxy = max(y3, y4);
      if (not(x >= minx and x <= maxx)) {
        cout << "NO\n";
        goto END;
      }
      if (not(y >= miny and y <= maxy)) {
        cout << "NO\n";
        goto END;
      }
      cout << "YES\n";
    }
  END:
    ++counter;
  }
}

void point_in_polygon() {
  st N, M;
  cin >> N;
  cin >> M;

  st counter = 0;
  int x, y;
  int vertx[N];
  int verty[N];
  vector<pair<int, int>> points;
  // XXX: The vertices of the polygon.
  while (counter < N) {
    cin >> x;
    cin >> y;
    vertx[counter] = x;
    verty[counter] = y;
    ++counter;
  }
  counter = 0;
  // XXX: All the points to check
  while (counter < M) {
    cin >> x;
    cin >> y;
    points.push_back({x, y});
    ++counter;
  }

  // XXX: Ray tracing test -- odd number of intersections is inside
  // point, else outside.
  // https://wrfranklin.org/Research/Short_Notes/pnpoly.html
  auto test = [&vertx, &verty, &N](float testx, float testy) -> bool {
    int i, j, c = 0;
    for (i = 0, j = N - 1; i < N; j = i++) {
      if (((verty[i] > testy) != (verty[j] > testy)) &&
          (testx <
           (vertx[j] - vertx[i]) * (testy - verty[i]) / (verty[j] - verty[i]) +
               vertx[i]))
        c = !c;
    }
    return c;
  };

  for (auto &[testx, testy] : points) {
    if (test(testx, testy))
      cout << "INSIDE\n";
    else
      cout << "OUTSIDE\n";
  }
}

void tree_dfs(const int node, int counter, const vector<int> *adj,
              vector<bool> &vis, int *plen) {
  vis[node] = true;
  plen[node] = counter;

  for (const int &child : adj[node]) {
    if (!vis[child]) {
      tree_dfs(child, counter + 1, adj, vis, plen);
    }
  }
}

void tree_lens(const vector<int> *adj, const int N, int *plens) {
  vector<bool> vis(N, false);
  int counter = 0;
  for (st i = 0; i < N; ++i) {
    int *plen = &plens[i * N];
    // XXX: A array to hold the counter for reachability to each node
    // from this node.
    fill(&plen[0], &plen[0] + N, 0);
    fill(begin(vis), end(vis), false);
    counter = 0;
    tree_dfs(i, counter, adj, vis, plen);
  }
}

void tree_diameter() {
  st N;
  cin >> N;
  vector<int> adj[N];
  st counter = 0;
  int s, d;
  while (counter < N - 1) {
    cin >> s;
    --s;
    cin >> d;
    --d;
    adj[s].push_back(d);
    adj[d].push_back(s);
    ++counter;
  }

  // print_iter(&adj[0], &adj[0] + N);

  // XXX: Now get the diameter of the treea using DFS
  int plens[N * N]; // O(N^2) space usage!
  vector<bool> vis(N, false);
  int diameter = 0;
  tree_lens(adj, N, plens);
  // 2*O(|V|^2)
  for (st i = 0; i < N; ++i) {
    int me = *max_element(&plens[i * N], &plens[i * N] + N);
    diameter = max(diameter, me);
  }
  cout << diameter << "\n";
}

void tree_dist_2() {
  st N;
  cin >> N;
  st counter = 0;
  vector<int> adj[N];
  int s, d;
  while (counter < N - 1) {
    cin >> s;
    --s;
    cin >> d;
    --d;
    adj[s].push_back(d);
    adj[d].push_back(s);
    ++counter;
  }
  counter = 0;
  int plens[N * N];
  tree_lens(adj, N, plens);
  for (st i = 0; i < N; ++i) {
    cout << accumulate(&plens[i * N], &plens[i * N] + N, 0) << " ";
  }
  cout << "\n";
}

void get_parents(const vector<int> *adj, vector<int> &p, const int c) {
  // XXX: Just one or none parent possible in a tree
  if (adj[c].size() != 0) {
    p.push_back(adj[c][0]);
    get_parents(adj, p, adj[c][0]);
  } else if (p.empty())
    // XXX: Adding yourself, since it was asked for!
    p.push_back(c);
}

void company_query_2() {
  st N, q;
  cin >> N;       // number of employees
  cin >> q;       // queries
  st counter = 1; // start from the first employee
  // XXX: This is a DAG going from child to parent only!
  vector<int> adj[N]; // only N-1 employees have a boss
  int b;
  while (counter < N) {
    cin >> b;
    --b;
    // XXX: Employee counter' boss is b
    adj[counter].push_back(b);
    ++counter;
  }
  counter = 0;
  // print_iter(&adj[0], &adj[0] + N);
  // XXX: Now read the queries
  vector<pair<int, int>> qs;
  int e1, e2;
  while (counter < q) {
    cin >> e1;
    --e1;
    cin >> e2;
    --e2;
    qs.push_back({e1, e2});
    ++counter;
  }
  // cout << qs;
  // XXX: Now get the least common parent from a given query!
  vector<int> p1, p2;
  for (auto &[q1, q2] : qs) {
    if (!p1.empty())
      p1.clear();
    if (!p2.empty())
      p2.clear();
    // XXX: Get all reachable parents from q1 and q2 in order using DFS
    // XXX: Note that this works, because we have made a DAG from the tree
    get_parents(adj, p1, q1);
    get_parents(adj, p2, q2);
    // XXX: Now we have the parents in order from both nodes just
    // traverse and get the first common parent in both vectors
    for (auto &x : p1) {
      auto it = find(begin(p2), end(p2), x);
      if (it != end(p2)) {
        cout << (*it) + 1 << "\n";
        break;
      }
    }
  }
}

int get_sum(const vector<int> *adj, vector<bool> &vis, int node,
            const int *vals) {
  vis[node] = true;
  int sum = vals[node];
  for (const int &x : adj[node]) {
    if (!vis[x]) {
      sum += get_sum(adj, vis, x, vals);
    }
  }
  return sum;
}

int get_tree_sum(const vector<int> *adj, const int *vals, vector<bool> &vis,
                 const int s, int r, bool &done) {
  int sum = 0;
  vis[r] = true;
  if (s == r) {
    // XXX: Found the node that I was searching for
    done = true;
    sum = get_sum(adj, vis, s, vals);
  } else
    for (const int &x : adj[r]) {
      if (!vis[x]) {
        sum = get_tree_sum(adj, vals, vis, s, x, done);
        if (done)
          break;
      }
    }
  return sum;
}

void sub_tree_query() {
  st N, Q;
  cin >> N;
  cin >> Q;
  st counter = 0;
  int node_vals[N]; // values in each node
  while (counter < N)
    cin >> node_vals[counter++];
  counter = 0;
  vector<int> adj[N];
  int s, d;
  while (counter < N - 1) {
    cin >> s;
    --s;
    cin >> d;
    --d;
    adj[s].push_back(d);
    adj[s].push_back(s);
    ++counter;
  }
  counter = 0;
  vector<pair<int, optional<int>>> qs;
  int q;
  while (counter < Q) {
    cin >> q;
    if (q == 1) {
      cin >> s;
      --s;
      cin >> d;
      qs.push_back({s, d});
    } else if (q == 2) {
      cin >> s;
      --s;
      qs.push_back({s, nullopt});
    }
    ++counter;
  }

  // XXX: Now start processing the queries
  vector<bool> vis(N, false);
  bool done = false;
  for (auto &[n, ov] : qs) {
    done = false;
    fill(begin(vis), end(vis), false);
    if (!ov) {
      // XXX: Get the sum from the subtree
      cout << get_tree_sum(adj, node_vals, vis, n, 0, done) << "\n";
    } else {
      // XXX: Change the value in the subtree
      node_vals[n] = *ov;
    }
  }
}

int get_max_in_path(const vector<int> *adj, const int *vals, int s, const int d,
                    vector<bool> &vis, bool &done) {
  int m = vals[s];
  vis[s] = true;
  if (s == d) {
    done = true;
  } else
    for (const int &x : adj[s]) {
      if (!vis[x]) {
        int m1 = get_max_in_path(adj, vals, x, d, vis, done);
        if (done) {
          m = max(m, m1);
          break;
        }
      }
    }
  return m;
}

void path_query_2() {
  st N, Q;
  cin >> N;
  cin >> Q;
  vector<int> adj[N];
  int vals[N];
  st counter = 0;
  while (counter < N)
    cin >> vals[counter++];
  counter = 0;
  // XXX: Note that in a tree, edges == |V|-1
  int s, d;
  while (counter < N - 1) {
    cin >> s;
    --s;
    cin >> d;
    --d;
    adj[s].push_back(d);
    adj[d].push_back(s);
    ++counter;
  }
  counter = 0;
  vector<tuple<int, int, int>> qs;
  int q;
  while (counter < Q) {
    cin >> q;
    cin >> s;
    --s;
    cin >> d;
    if (q == 2)
      --d;
    qs.push_back({q, s, d});
    ++counter;
  }
  counter = 0;
  // XXX: Process the queries
  vector<bool> vis(N, false);
  bool done = false;
  for (auto &[q, s, d] : qs) {
    fill(begin(vis), end(vis), false);
    done = false;
    if (q == 1) {
      vals[s] = d;
    } else if (q == 2) {
      // XXX: Traverse the tree from the given node to the destination node and
      // get the max value in the path
      cout << get_max_in_path(adj, vals, s, d, vis, done) << " ";
    }
  }
  cout << endl;
}

int get_tree_size(const vector<int> *adj, vector<bool> &vis, int r) {
  vis[r] = true;
  int sum = 1;
  for (const int &x : adj[r]) {
    if (!vis[x]) {
      sum += get_tree_size(adj, vis, x);
    }
  }
  return sum;
}

int tree_centroid(const vector<int> *adj, const int N, vector<bool> &vis,
                  int r) {
  st i = 0;
  vis[r] = true;
  int sum = numeric_limits<int>::max();
  for (const int &x : adj[r]) {
    if (!vis[x]) {
      // XXX: Get the sum for this subtree
      sum = get_tree_size(adj, vis, x);
      // cout << "r: " << r << "\n";
      // cout << "sum is: " << sum << "\n";
      if (sum > N / 2) {
        break;
      }
      ++i;
    }
  }
  // cout << "i: " << i << "\n";
  if (sum <= N / 2) {
    return r;
  } else {
    fill(begin(vis), end(vis), false);
    return tree_centroid(adj, N, vis, adj[r][i]);
  }
}

void centroid() {
  st N;
  cin >> N;
  st counter = 0;
  vector<int> adj[N];
  int s, d;
  while (counter < N - 1) {
    cin >> s;
    --s;
    cin >> d;
    --d;
    adj[s].push_back(d);
    adj[d].push_back(s);
    ++counter;
  }
  // print_iter(&adj[0], &adj[0] + N);
  // XXX: Now find the centroid
  vector<bool> vis(N, false);
  cout << tree_centroid(adj, N, vis, 0) + 1 << "\n";
}

// XXX: Can be made faster with O(log N), space (4N) with segment tree
void static_range_sum() {
  st N, Q;
  cin >> N;
  cin >> Q;
  int vals[N];
  st counter = 0;
  while (counter < N) {
    cin >> vals[counter++];
  }
  counter = 0;
  int s, e;
  while (counter < Q) {
    cin >> s;
    --s;
    cin >> e;
    --e;
    cout << accumulate(&vals[0] + s, &vals[0] + e + 1, 0) << "\n";
    ++counter;
  }
}

// XXX: Can be made faster O(log N), but spaec (4N) using segment tree
void static_range_min() {
  st N, Q;
  cin >> N;
  cin >> Q;
  int vals[N];
  st counter = 0;
  while (counter < N) {
    cin >> vals[counter++];
  }
  counter = 0;
  int s, e;
  while (counter < Q) {
    cin >> s;
    --s;
    cin >> e;
    --e;
    cout << *min_element(&vals[0] + s, &vals[0] + e + 1) << "\n";
    ++counter;
  }
}

// XXX: Can be made faster with segment tree O(log N), space (4N)
void xor_sum() {
  st N, Q;
  cin >> N;
  cin >> Q;
  int vals[N];
  st counter = 0;
  while (counter < N) {
    cin >> vals[counter++];
  }
  counter = 0;
  int s, e;
  while (counter < Q) {
    cin >> s;
    --s;
    cin >> e;
    --e;
    cout << accumulate(&vals[0] + s, &vals[0] + e + 1, 0, [](int x, int y) {
      return x ^ y;
    }) << "\n";
    ++counter;
  }
}

// XXX: Can be made faster with O(log N), space (4N)
void range_update_query() {
  st N, Q;
  cin >> N;
  cin >> Q;
  int vals[N];
  st counter = 0;
  while (counter < N) {
    cin >> vals[counter++];
  }
  counter = 0;
  int s, e, q, u;
  while (counter < Q) {
    cin >> q;
    if (q == 1) {
      cin >> s;
      --s;
      cin >> e;
      --s;
      cin >> u;
      // XXX: Transform the array
      transform(&vals[0] + s, &vals[0] + e + 1, &vals[0] + s,
                [&u](int x) { return x + u; });
    } else if (q == 2) {
      cin >> s;
      --s;
      cout << vals[s] << "\n";
    }
    ++counter;
  }
}

// XXX: Used for generic base operator for projecting the value.
template <typename T> struct BOP {
  T operator()(T v, st l, st r) const { return v; }
};

// XXX: Segment tree index to update. See query_seg_tree_by_indices
// function how to get the segment tree index for a given array index.
// Complexity O(log N)
template <typename T>
void update_seg_tree_by_index(const st index, const T &val,
                              tuple<T, st, st> *res,
                              const auto &op = plus<T>{}) {
  // XXX: Update the leaf node.
  auto &[v, l, r] = res[index];
  res[index] = {val, l, r};
  // XXX: Now update the parent.
  st counter = index;
  while (counter != 0) {
    // XXX: Get the parent and update its value
    if (counter % 2 == 0) {
      // XXX: Even index
      counter = (counter - 2) / 2;
    } else {
      counter = (counter - 1) / 2;
    }
    auto &[_, l, r] = res[counter];
    auto &[v1, l1, r1] = res[counter * 2 + 1];
    auto &[v2, l2, r2] = res[counter * 2 + 2];
    res[counter] = {op(v1, v2), l, r};
  }
}

// XXX: Gives back the seg tree value found between lindex and rindex of
// the original array indices. All indices start from 0. Complexity:
// O(log N).
template <typename T, class F = plus<T>>
T query_seg_tree_by_indices(const st lindex, const st rindex,
                            const tuple<T, st, st> *res, const F &op = F{},
                            int counter = 0) {
  assert(lindex <= rindex);
  auto &[v, l, r] = res[counter];
  assert(lindex >= l and rindex <= r);
  if ((lindex == l) and (rindex == r)) {
    return v;
  } else if (rindex <= get<2>(res[2 * counter + 1])) {
    return query_seg_tree_by_indices(lindex, rindex, res, (2 * counter + 1));
  } else if (lindex >= get<1>(res[2 * counter + 2])) {
    return query_seg_tree_by_indices(lindex, rindex, res, (2 * counter + 2));
  } else {
    // XXX: The most complex case search both branches
    T r1 = query_seg_tree_by_indices(lindex, get<2>(res[counter * 2 + 1]), res,
                                     counter * 2 + 1);
    T r2 = query_seg_tree_by_indices(get<2>(res[counter * 2 + 1]) + 1, rindex,
                                     res, counter * 2 + 2);
    return op(r1, r2);
  }
}

// XXX: Gives back the seg tree index for the given orignal array index.
// Complexity: O(log N)
template <typename T>
st query_seg_tree_by_indices(const st index, const tuple<T, st, st> *res,
                             int counter = 0) {
  auto &[v, l, r] = res[counter];
  assert(index >= l and index <= r);
  if ((index == l) and (index == r)) {
    return counter;
  } else if (index <= get<2>(res[2 * counter + 1])) {
    return query_seg_tree_by_indices(index, res, (2 * counter + 1));
  } else if (index >= get<1>(res[2 * counter + 2])) {
    return query_seg_tree_by_indices(index, res, (2 * counter + 2));
  } else {
    // XXX: Can never reach here!
    assert(false);
  }
}

// XXX: It will return: (1) value in the node, (2) index in the
// original array, and (3) index in the segment tree. Indices will be
// -1 if the value is not found in the seg tree and val will be
// returned. Complexity: O(log N).
template <typename T, class F = less_equal<T>>
tuple<T, int, int> query_seg_tree_by_value(const T &val, tuple<T, st, st> *res,
                                           st counter = 0, const F &qop = F{}) {
  const auto &[v, l, r] = res[counter];
  if (qop(val, v) and (l == r)) {
    return {v, l, counter};
  } else if (qop(val, v) and (l < r)) {
    // XXX: First check the left child
    auto &&[vv, r, ti] = query_seg_tree_by_value(val, res, (2 * counter + 1));
    if (r != -1)
      return {vv, r, ti};
    else
      // XXX: Then check the right child
      return query_seg_tree_by_value(val, res, (2 * counter + 2));
  } else
    return {val, -1, -1};
}

template <typename T, typename U, class F = plus<T>, class Y = BOP<U>>
void build_seg_tree_with_index(const U *const B, int counter, U *b, U *e,
                               tuple<T, st, st> *res, const F &op = F{},
                               const Y &bop = Y{}) {
  // XXX: Note that F is a struct plus<T>, op is object of that struct type with
  // operator(). Note that auto does not work with default operator types!
  if ((e - b) == 1) {
    // XXX: Reached the end of the segment tree
    res[counter] = {bop(*b, (b - B), (e - B - 1)), (b - B), (e - B - 1)};
  } else {
    st mid = ((e - b) / 2);
    // XXX: Make the left segment tree
    build_seg_tree_with_index(B, (2 * counter + 1), b, b + mid, res, op, bop);
    // XXX: Make the right segment tree
    build_seg_tree_with_index(B, (2 * counter + 2), b + mid, e, res, op, bop);
    auto &[res1, i1, i2] = res[(2 * counter + 1)];
    auto &[res2, i11, i22] = res[(2 * counter + 2)];
    res[counter] = {op(res1, res2), (b - B), (e - B - 1)};
  }
}

void hotel_query() {
  st N, M;
  cin >> N; // number of hotels (array size)
  cin >> M; // array size of groups.
  st counter = 0;
  int hotels[N];
  while (counter < N) {
    cin >> hotels[counter++];
  }
  counter = 0;
  int groups[M];
  while (counter < M) {
    cin >> groups[counter++];
  }
  // print_iter(&hotels[0], &hotels[0] + N);
  // print_iter(&groups[0], &groups[0] + M);

  // XXX: Allocate space for the segment tree.
  st h = static_cast<st>(ceil(log2(N)));
  st size = 0;
  for (st i = 0; i <= h; ++i)
    size += (st)pow(2, i);
  tuple<int, st, st> seg_tree[size];
  // XXX: Note that mmax is actually a struct with operator(T a, T b) -> T
  auto mmax = [](int x, int y) { return max(x, y); };
  build_seg_tree_with_index(&hotels[0], 0, &hotels[0], &hotels[0] + N,
                            &seg_tree[0], mmax);
  // print_iter(&seg_tree[0], &seg_tree[0] + size);

  // XXX: Now make a query to in the seg tree to get the index with the given
  // value
  for (const int &gv : groups) {
    auto &&[v, ai, ti] = query_seg_tree_by_value(gv, &seg_tree[0], 0);
    // cout << "v: " << v << ", ai: " << ai << ", ti: " << ti << "\n";
    // XXX: Update the value in the seg tree
    if (ti != -1) {
      int uv = v - gv;
      update_seg_tree_by_index(ti, uv, seg_tree, mmax);
    }
    cout << ai + 1 << " "; // incrementing, because everything starts with 1
  }
  cout << "\n";
}

// XXX: This is easiest implemented as O(N), there is O(log N) technique
// with segment tree, but unclear right now how to handle, overlapping
// prices.
void pizza_query() {
  st N, Q;
  cin >> N;
  cin >> Q;
  st counter = 0;
  int prices[N];
  while (counter < N) {
    cin >> prices[counter++];
  }
  counter = 0;
  int q, i, me, j, res, p;
  while (counter < Q) {
    cin >> q;
    if (q == 1) {
      // XXX: Just update the value
      cin >> i;
      --i;
      cin >> prices[i];
    } else if (q == 2) {
      // XXX: Just do a O(N) algorithm
      cin >> me;
      --me;
      j = 0;
      res = numeric_limits<int>::max();
      for (int &x : prices) {
        p = x + abs(j - me);
        res = min(res, p);
        ++j;
      }
      cout << res << "\n";
    }
    ++counter;
  }
}

void subarray_max_sum() {
  st N, M;
  st counter = 0;
  cin >> N;
  cin >> M;
  int array[N];
  pair<int, int> qs[M];
  while (counter < N) {
    cin >> array[counter++];
  }
  counter = 0;
  int index, val;
  while (counter < M) {
    cin >> index;
    --index;
    cin >> val;
    qs[counter++] = {index, val};
    // XXX: Get the seg tree index
  }
  using s_t = tuple<int, int, int, int>;
  // XXX: Build the segment tree with the given structure and operation.
  auto bop = [](int v, st x, st y) { return s_t{v, v, v, v}; };
  // XXX: Merge between 2 nodes
  auto op = [](s_t x, s_t y) {
    auto &[xts, xps, xss, xms] = x;
    auto &[yts, yps, yss, yms] = y;
    int ts = xts + yts;
    int ps = max(xps, (xts + yps));
    int ss = max(yss, (yts + xss));
    int ms = max({xms, yms, (xss + yps)});
    return s_t{ts, ps, ss, ms};
  };
  // XXX: Build the segment tree
  // XXX: Allocate space for the segment tree.
  st h = static_cast<st>(ceil(log2(N)));
  st size = 0;
  for (st i = 0; i <= h; ++i)
    size += (st)pow(2, i);
  tuple<s_t, st, st> seg_tree[size];
  build_seg_tree_with_index(array, 0, array, array + N, seg_tree, op, bop);
  // print_iter(seg_tree, seg_tree + size);
  for (const auto &[i, v] : qs) {
    // cout << "i: " << i << ", v: " << v << "\n";
    st si = query_seg_tree_by_indices(i, seg_tree, 0);
    update_seg_tree_by_index(si, {v, v, v, v}, seg_tree, op);
    const auto &[ts, _1, _2] = seg_tree[0];
    cout << get<3>(ts) << "\n";
  }
}

// XXX: I am doing this in O(n) time. However, you can do O(log n) time
// if you just increment the value of the correct node, in the segment
// tree, with index [l, r] given by the value n(n+1)/2, where n = (r -
// l) +1. In the worst case then the updating the tree would be O(log n)
// time and the result would be available in the top. Note, that O(n)
// using iota and accumulate lets one use SIMD.
void polynomial_queries() {
  st N, Q;
  cin >> N;
  cin >> Q;
  int arr[N];
  st counter = 0;
  while (counter < N)
    cin >> arr[counter++];
  // print_iter(arr, arr + N);
  int q, s, e;
  counter = 0;
  while (counter < Q) {
    cin >> q;
    cin >> s;
    --s;
    cin >> e;
    --e;
    if (q == 2) {
      // XXX: The +1 is needed in accumulate, because of [) semantics of
      // accumulate.
      cout << accumulate(arr + s, arr + e + 1, 0) << "\n";
    } else if (q == 1) {
      // XXX: Update the value of each element in the range [s, e+1) by
      // 1,2,...n
      for (st i = s; i <= (e - s); ++i) {
        arr[i] += (i + 1);
      }
    }
    ++counter;
  }
}

// XXX: I cannot compile this with Clang++ > -O0, because it gives the wrong
// output!
void task_assignment() {
  st N;
  cin >> N;
  // XXX: Make NxN matrix for allocation
  int A[N * N];
  int orig[N * N];
  st counter = 0;
  int i = 0;
  while (counter < N) {
    i = 0;
    while (i < N) {
      cin >> A[N * counter + i];
      orig[N * counter + i] = A[N * counter + i];
      i++;
    }
    counter += 1;
  }
  // print_iter(A, A + (N * N));

  // XXX: This is a balanced minimisation problem, so no preprocessing
  // needed see:
  // https://www.cs.emory.edu/~cheung/Courses/253/Syllabus//Assignment/algorithm.html
  // https://kanchiuniv.ac.in/coursematerials/OperationResearch.pdf

  // XXX: Steps:

  // 1. First identify the lowest edges from x -> y, make then 0.
  // subtract lowest from the row 2. Next identify the lowest edges from
  // y -> x, make them 0. Subtract lowest from column. 3. Make a new
  // subgraph with lowest edges. Do maximal matching (max-flow karp) for
  // this subgraph. If maximal matching contains all xs then stop
  // solution found. If not, then Step 4. Add more edges the min edge of
  // unmatched to y' with no edge to them yet. Subtract the lowest value
  // from all rows of matched -> no edge y's. Add the min to all matched
  // x -> y's . Step 4 addition/subtraction only for cost > 0. Do
  // Step 3 and 4 iteratively until maximum matching is found. O(N^4)
  // complexity algorithm.

  const int s = (2 * N); // index of source
  const int t = s + 1;   // index of target
  const int row_size = t + 1;
  const int col_size = t + 1;

  // for (st i = 0; i < N; ++i) {
  //   cout << i << ": ";
  //   print_iter(&A[i * N], &A[i * N] + N);
  // }
  // cout << "\n";
  // XXX: Step-1
  // O(N^2)
  int mine = INT_MAX;
  for (st i = 0; i < N; ++i) {
    // XXX: get min element in the row
    mine = INT_MAX;
    mine = *min_element(&A[i * N], &A[i * N + N]);
    transform(&A[i * N], &A[i * N + N], &A[i * N],
              [&mine](int x) { return (x - mine); });
  }
  // for (st i = 0; i < N; ++i) {
  //   cout << i << ": ";
  //   print_iter(&A[i * N], &A[i * N] + N);
  // }
  // cout << "\n";

  // XXX: Step-2 do the same for the column
  // O(N^2)
  mine = INT_MAX;
  for (st j = 0; j < N; ++j) {
    mine = INT_MAX;
    for (st i = 0; i < N; ++i)
      mine = min(mine, A[i * N + j]);
    for (st i = 0; i < N; ++i)
      A[i * N + j] -= mine;
  }
  // for (st i = 0; i < N; ++i) {
  //   cout << i << ": ";
  //   print_iter(&A[i * N], &A[i * N] + N);
  // }
  // cout << "\n";

  // XXX: Now make the adjacency list format for the zero elements
  vector<int> parents(row_size, -1); // O(N)
  int capacity[row_size * col_size]; // O(N^2)
  vector<int> adj[row_size];         // 2 extra for source and target
  auto make_cap_adj = [&]() {
    fill(capacity, capacity + row_size * col_size, -1);
    for (st i = 0; i < N; ++i) {
      for (st j = 0; j < N; ++j) {
        if (!A[i * N + j]) {
          adj[i].push_back(j + N);
          adj[j + N].push_back(i);
          // XXX: Add the source to the left nodes
          if (find(adj[s].begin(), adj[s].end(), i) == adj[s].end())
            adj[s].push_back(i);
          if (find(adj[i].begin(), adj[i].end(), s) == adj[i].end())
            adj[i].push_back(s);
          // XXX: Add the right nodes to the target
          if (find(adj[t].begin(), adj[t].end(), (j + N)) == adj[t].end())
            adj[t].push_back(j + N);
          if (find(adj[j + N].begin(), adj[j + N].end(), t) == end(adj[j + N]))
            adj[j + N].push_back(t);

          // XXX: Make capacity matrix values
          capacity[i * row_size + (j + N)] = 1;
          capacity[(j + N) * row_size + i] = 0;
          capacity[s * row_size + i] = 1;
          capacity[i * row_size + s] = 0;
          capacity[(j + N) * row_size + t] = 1;
          capacity[t * row_size + (j + N)] = 0;
        }
      }
    }
  };
  make_cap_adj();
  // for (st i = 0; i < N; ++i) {
  //   cout << i << ": ";
  //   print_iter(&A[i * N], &A[i * N] + N);
  // }
  // print_iter(adj, adj + row_size);
  // for (int i = 0; i < row_size; ++i) {
  //   cout << i << ": ";
  //   print_iter(&capacity[i * row_size], &capacity[i * row_size] + col_size);
  // }
  // XXX: Now perform the max-flow
  int flow = -1;
  int total = 0;
  vector<tuple<int, int>> matches;
  auto matching = [&]() {
    flow = -1;
    total = 0;
    matches.clear();
    while (1) {
      flow = bfs_max_flow(s, row_size, adj, capacity, parents, t);
      if (!flow)
        break;
      int c = t;
      int p;
      while (c != s) {
        // XXX: Update the capacity matrix
        p = parents[c];
        capacity[p * row_size + c] -= flow;
        capacity[c * row_size + p] += flow;
        c = p;
      }
    }
    // XXX: Get the matchings and check if it is a maximum matching
    for (st i = 0; i < N; ++i) {
      for (st j = 0; j < N; ++j) {
        if (!capacity[i * row_size + (j + N)]) {
          // XXX: Found a match
          matches.push_back({i, j});
          total += orig[i * N + j];
        }
      }
    }
  };
  matching(); // Works for the given example.
  vector<int> unmatched;
  while (matches.size() != N) {
    unmatched.clear();
    // O(N^2)
    for (st i = 0; i < N; ++i) {
      if (find_if(matches.begin(), matches.end(),
                  [&i](const tuple<int, int> &p) { return get<0>(p) == i; }) ==
          matches.end()) {
        unmatched.push_back(i);
      }
    }
    vector<int> unmatched_adj[unmatched.size()];
    // XXX: First get the min element from all unmatched --> no connection edges
    int delta = INT_MAX; // This is delta
    for (int i : unmatched) {
      for (st j = 0; j < N; ++j) {
        if (A[i * N + j] > 0) {
          // XXX: there is no edge between this job and i
          delta = min(delta, A[i * N + j]);
        } else if (A[i * N + j] == 0)
          unmatched_adj[i].push_back(j);
      }
    }
    // XXX: Now subtract the delta from all these edges
    for (int i : unmatched) {
      for (st j = 0; j < N; ++j) {
        if (A[i * N + j] > 0) {
          A[i * N + j] -= delta;
        }
      }
    }
    // XXX: Add to other edges (CHECK!)
    for (auto &[i, _] : matches) {
      for (int u : unmatched) {
        for (int k : unmatched_adj[u]) {
          if (A[i * N + k] > 0)
            A[i * N + k] += delta;
        }
      }
    }
    // XXX: Iterate
    make_cap_adj();
    matching();
  }
  // XXX: If the matches size == N, the solution found
  cout << total << "\n";
  transform(matches.begin(), matches.end(), matches.begin(),
            [](const tuple<int, int> &p) -> tuple<int, int> {
              auto &[x, y] = p;
              return {x + 1, y + 1};
            });
  for (auto &[x, y] : matches) {
    cout << x << " " << y << "\n";
  }
}

int main() {
  // XXX: Sorting and searching algo
  // list_to_set();
  // gray_code();
  // sum_of_two();// O(N) time, with hash-table
  // fwheel();
  // ad_sup();
  // slide_median();

  // XXX: Graph algorithms
  // shortest_route_1(); //Dijkstra
  // course_schedule(); // topological sort
  // road_reparation(); //Minimum spanning tree
  // road_construction();// minimum spaning tree, disjoint union of sets
  // flight_route_reqs();// reachability
  // forbidden_cities(); //reachability
  // new_flight_routes();// strongly connected components, Kosaraju algo
  // giant_pizza(); //2-SAT problem, implication graph, Kosaraju
  // grid_search();//Hamiltonian path, dynamic programming
  // police_chase();//Max network flow, min st cut, Karp algo
  // school_dance();//Max flow, matching bipartite graph, Karp algo
  // teleporters_path();//Eulerian path.
  // building_teams(); // Bipartite graph colouring

  // XXX: Geometry
  // point_line_location(); // (left, right or touch)
  // line_segment_intersection();  //intersection of lines and then seg check
  // min_euclid_dist(); //a sorted distance find
  // polygon_area();//Shoelace (general), heron (convex)
  // point_in_polygon();//ray intersection with sides of polygon
  // TODO: Later
  // convex_hull();//graham scan

  // XXX: Tree algorithms (dfs: O(|V|)), because |E| = |V|-1 in a tree
  // tree_diameter();
  // tree_dist_2();
  // company_query_2(); //Note that this can be done by inverting the tree
  // sub_tree_query();
  // path_query_2();
  // centroid(); // Imp

  // XXX: Range query algorithms (segment trees)
  // static_range_sum();
  // static_range_min();
  // xor_sum();
  // range_update_query();
  // hotel_query(); //easy segment tree
  // pizza_query();
  // Very important; what is max prefix, max suffix, and max sub-array sum
  // subarray_max_sum();
  // polynomial_queries();

  // XXX: Task assignment problem for NxN task allocation problem in
  // polynomial time! (Hungarian algorithm) -- max-flow extension
  task_assignment();

  // XXX: Test for a recursive lambda in 2 different ways

  // XXX: String algorithms

  return 0;
}
