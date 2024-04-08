#include <cstddef>
#include <functional>
#include <iostream>
#include <numeric>
#include <ranges>
#include <string>
#include <vector>

// XXX: Search one given character in a string
void char_pos(const std::string s, const char c, std::vector<size_t> &vec) {
  for (size_t i = 0; i < s.size(); ++i) {
    if (toupper(s[i]) == toupper(c)) {
      vec.push_back(i);
    }
  }
}

// XXX: Count the number of char in the string
size_t count_cs(const std::string s, const char c) {
  size_t toret = 0;
  for (const char x : s) {
    if (toupper(x) == toupper(c))
      ++toret;
  }
  return toret;
}

void char_pos_algo(const std::string s, const char c,
                   std::vector<size_t> &vec) {
  for (int _ :
       std::ranges::views::iota(0) | std::ranges::views::take_while([&](int x) {
         if (toupper(s[x]) == toupper(c)) {
           vec.push_back(x);
         }
         return x < s.size();
       }))
    ;
}

int main() {
  // XXX: The problem is to search characters in a string
  std::string s = "Hello World!";
  std::vector<size_t> vec;
  const char c = 'l';
  char_pos(s, c, vec);
  for (size_t p : vec) {
    std::cout << p << ' ';
  }
  std::cout << "\n";

  // XXX: Using algorithms
  vec.clear();
  char_pos_algo(s, c, vec);
  for (size_t p : vec) {
    std::cout << p << ' ';
  }
  std::cout << "\n";

  // XXX: Get the number of characters in string
  std::cout << count_cs(s, c) << "\n";

  // XXX: The same thing with the algorithm library
  std::cout << std::ranges::count_if(s, [](const char x) {
    return (toupper(x) == toupper(c));
  }) << "\n";

  // XXX: Join strings in C++
  std::vector<std::string> vs{"Hello", "How", "Are", "You", "?"};

  // XXX: Join string using the standard loop
  std::string joined;
  for (std::string &x : vs) {
    joined += x + " ";
  }
  std::cout << joined << "\n";

  // XXX: Join using algorithms
  std::string j2;
  auto add_space = [](std::string x) { return x + " "; };
  std::cout << std::transform_reduce(vs.begin(), vs.end(), std::string{""},
                                     std::plus<std::string>{}, add_space)
            << "\n";

  return 0;
}
