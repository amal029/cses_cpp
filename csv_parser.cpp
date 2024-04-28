#include <iostream>
#include <ranges>
#include <string>
#include <vector>

using namespace std;

template <typename T> ostream &operator<<(ostream &os, vector<T> vec) {
  for (const auto &x : vec) {
    cout << x << " ";
  }
  cout << "\n";
  return os;
}

string join(vector<string> s) {
  string toret = "";
  for (const string &x : s) {
    toret += x + " ";
  }
  return toret;
}

void dec_csv_parse(const char &&sep = ',') {
  string s;
  auto f1 = [&sep](char x, char y) { return not(x == sep or y == sep); };
  // auto f2 = [&sep](auto &z) { return z != string{sep}; };
  string t;
  while (cin >> s) {
    t.clear();
    auto vv = s | ranges::views::chunk_by(f1);
    for (auto const &x : vv) {
      for (const char &e : x) {
        if (e != sep) {
          t += e;
        }
      }
      cout << t << " ";
      t.clear();
    }
    cout << "\n";
  }
}

void parse_csv(char &&sep = ',') {
  string s;
  vector<string> ss;
  int pos = 0;
  int ppos = pos;
  while (cin >> s) {
    pos = ppos = 0;
    pos = s.find(sep, pos);
    while (pos != string::npos) {
      ss.push_back(s.substr(ppos, pos - ppos));
      ppos = pos + 1;
      pos = s.find(sep, ppos);
    }
    // XXX: One last time
    ss.push_back(s.substr(ppos));
    cout << join(ss) << endl;
    ss.clear();
  }
}

int main(int argc, char *argv[]) {
  if (argc == 1)
    dec_csv_parse(',');
  else {
    dec_csv_parse(std::move(*argv[1]));
  }
  // if (argc == 1)
  //   parse_csv(',');
  // else {
  //   parse_csv(std::move(*argv[1]));
  // }
  return 0;
}
