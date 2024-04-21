#include <charconv>
#include <cstring>
#include <iostream>
#include <ostream>
#include <string.h>
#include <utility>

using namespace std;

class mstring {
public:
  mstring() : s(nullptr), n(0) {}
  mstring(const char *c) {
    std::cout << "Called constructor \n";
    s = new char[strlen(c) + 1];
    strcpy(s, c);
    n = strlen(s) + 1;
  };
  // XXX: Copy constructor
  mstring(const mstring &c) {
    cout << "Called copy constructor\n";
    s = new char[c.n];
    // XXX: This is slower than memcpy, but address sanitizer complains
    // if one uses memcpy.
    strcpy(s, c.s);
    n = c.n;
  };
  // XXX: Assignment operator
  mstring &operator=(const mstring &c) {
    // XXX: The input has to be via value-copy
    cout << "Called copy assignment\n";
    // XXX: Make a temp first
    char *temp = new char[c.n];
    strcpy(temp, c.s);
    swap(s, temp);
    n = c.n;
    delete[] temp;
    return *this;
  }

  // XXX: Move constructor
  mstring(mstring &&c) {
    cout << "Called move constructor\n";
    swap(s, c.s);
    n = c.n;
    c.s = nullptr;
    c.n = 0;
  }

  // XXX: Move assignment operator
  mstring &operator=(mstring &&x) {
    cout << "Called move assignment" << endl;
    delete[] s;
    swap(s, x.s);
    n = x.n;
    x.s = nullptr;
    return *this;
  }

  virtual ~mstring() {
    cout << "Called destructor\n";
    delete[] s;
  }

  // XXX: This needs to be a friend.
  friend std::ostream &operator<<(std::ostream &, const mstring &);

private:
  char *s;
  size_t n;
};

std::ostream &operator<<(std::ostream &of, const mstring &m) {
  for (size_t i = 0; i < m.n - 1; ++i) {
    of << m.s[i];
  }
  return of;
}

int main() {
  mstring s{"Hello"};
  std::cout << s << "\n";
  mstring s1{s};
  cout << s1 << "\n";
  mstring s2;
  s2 = s1;
  cout << "s2:" << s2 << "\n";
  mstring s3{"PIPIPIPIPIPI"};
  cout << s3 << "\n";
  s2 = std::move(s3);
  cout << s2 << "\n";
  return 0;
}
