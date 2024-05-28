#include <iostream>
#include <variant>

// XXX: This is the interface
class State {};

// XXX: These are my states
class S0 : State {};
class S1 : State {};
class S2 : State {};

// XXX: This is the thread with data
template <class State> class Thread {};

// XXX: Make a variant of thread state
using ThreadState = std::variant<Thread<S0>, Thread<S1>, Thread<S2>>;

// XXX: Thread data
struct ThreadData {
  int a = 0;
  friend std::ostream &operator<<(std::ostream &, const ThreadData &);
};

std::ostream &operator<<(std::ostream &s, const ThreadData &d) {
  s << d.a << " ";
  return s;
}

// XXX: Specialise the Thread
template <> class Thread<S0> {
  S0 _c;

public:
  Thread(S0 &&c) : _c(c) {}
  friend std::ostream &operator<<(std::ostream &s, const Thread<S0> &);
  ThreadState tick(ThreadData &d);
};
template <> class Thread<S1> {
  S1 _c;

public:
  Thread(S1 &&c) : _c(c) {}
  friend std::ostream &operator<<(std::ostream &s, const Thread<S0> &);
  ThreadState tick(ThreadData &d);
};

template <> class Thread<S2> {
  S2 _c;

public:
  Thread(S2 &&c) : _c(c) {}
  friend std::ostream &operator<<(std::ostream &s, const Thread<S0> &);
  ThreadState tick(ThreadData &d);
};

std::ostream &operator<<(std::ostream &s, const Thread<S0> &sd) {
  s << "State S0"
    << " ";
  return s;
}

std::ostream &operator<<(std::ostream &s, const Thread<S1> &sd) {
  s << "State S1"
    << " ";
  return s;
}

std::ostream &operator<<(std::ostream &s, const Thread<S2> &sd) {
  s << "State S2"
    << " ";
  return s;
}

// XXX: Return by move
ThreadState Thread<S0>::tick(ThreadData &d) {
  if (d.a == 0) {
    d.a++;
    return S1{S1{}};
  } else {
    d.a--;
    return S2{S2{}};
  }
}

// XXX: Return by move
ThreadState Thread<S1>::tick(ThreadData &d) {
  d.a++;
  return S0{S0{}};
}

ThreadState Thread<S2>::tick(ThreadData &d) {
  d.a--;
  return S0{S0{}};
}

// XXX: The initial state. This is a return by move!
ThreadState init() { return S0{S0{}}; }

// XXX: Visit and call the tick function for the thread
template <class... Ts> struct overloaded : Ts... {
  using Ts::operator()...;
};
ThreadState visit(ThreadState &&ts, ThreadData &d) {
  return std::visit(overloaded{[&d](auto &t) { return t.tick(d); }}, ts);
}

void print_visit(const ThreadState &ts, const ThreadData &d) {
  return std::visit(overloaded{[&d](auto &t) { std::cout << t << d << "\n"; }},
                    ts);
}

int main(int argc, char *argv[]) {
  ThreadData data{};
  auto v = init();
  print_visit(v, data);
  for (int i = 0; i < 10; ++i) {
    // XXX: This assignment happens by copy, but why??
    v = visit(std::move(v), data);
    print_visit(v, data);
  }
  return 0;
}
