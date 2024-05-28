#include <iostream>
#include <variant>

// XXX: This is the interface
class State {};

// XXX: These are my states
class S0 : State {};
class S1 : State {};

// XXX: This is the thread with data
template <class State> class Thread {};

// XXX: Make a variant of thread state
using ThreadState = std::variant<Thread<S0>, Thread<S1>>;

// XXX: Specialise the Thread
template <> class Thread<S0> {
  S0 _c;

public:
  Thread(S0 &&c) : _c(c) {}
  constexpr Thread<S0> &operator=(const Thread<S0> &s) noexcept {
    std::cout << "Copy assignment S0\n";
    _c = s._c;
    return *this;
  };
  // constexpr Thread<S0> &operator=(Thread<S0> &&s) noexcept {
  //   std::swap(_c, s._c);
  //   return *this;
  // };
  ThreadState tick();
};
template <> class Thread<S1> {
  S1 _c;

public:
  Thread(S1 &&c) : _c(c) {}
  constexpr Thread<S1> &operator=(const Thread<S1> &s) noexcept {
    std::cout << "Copy assignment S1\n";
    _c = s._c;
    return *this;
  };
  // constexpr Thread<S1> &operator=(Thread<S1> &&s) noexcept {
  //   std::swap(_c, s._c);
  //   return *this;
  // };
  ThreadState tick();
};

// XXX: Return by copy
ThreadState Thread<S0>::tick() {
  std::cout << "S0 tick\n";
  return S1{S1{}};
}

// XXX: Return by copy
ThreadState Thread<S1>::tick() {
  std::cout << "S1 tick\n";
  return S0{S0{}};
}

// XXX: This is a return by move!
ThreadState init() { return S0{S0{}}; }

template <class... Ts> struct overloaded : Ts... {
  using Ts::operator()...;
};

ThreadState visit(ThreadState &ts) {
  return std::visit(overloaded{[](auto &t) { return t.tick(); }}, ts);
}

int main(int argc, char *argv[]) {
  auto v = init();
  for (int i = 0; i < 10; ++i) {
    // XXX: This assignment happens by copy, but why??
    v = visit(v);
  }
  return 0;
}
