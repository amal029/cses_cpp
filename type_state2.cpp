#include <iostream>
#include <variant>

struct signal_A {
  bool status;
};

struct signal_B {
  bool status;
  int64_t value;
  signal_B &operator+(signal_B &b) {
    this->value = this->value + b.value;
    return *this;
  }
};

// XXX: The objects of signal types
static signal_A curr_A, pre_A;
static signal_B curr_B, pre_B;

// XXX: Now all the states in the program
struct State {};
struct E : State {};
struct I : State {};
struct S0 : State {};
struct S1 : State {};
struct S2 : State {};
struct S4 : State {};
struct S3 : State {};
struct Th1ND : State {};
struct Th2ND : State {};
struct Th1D : State {};
struct Th2D : State {};
struct Th0ND : State {};

// XXX: Now define all the threads in the program
template <class State> struct Thread0 {};
template <class State> struct Thread1 {};
template <class State> struct Thread2 {};

// XXX: Define the threadstate variant
using Thread0State =
    std::variant<Thread0<I>, Thread0<S0>, Thread0<E>, Thread0<Th0ND>>;
using Thread1State = std::variant<Thread1<S1>, Thread1<S4>, Thread1<Th1ND>,
                                  Thread1<Th1D>, Thread1<I>>;
using Thread2State = std::variant<Thread2<S2>, Thread2<S3>, Thread2<Th2ND>,
                                  Thread2<Th2D>, Thread2<I>>;

// XXX: Now define all the prototypes
template <> struct Thread0<I> {
  constexpr void tick(signal_A &, signal_B &);
};
template <> struct Thread0<S0> {
  constexpr void tick(signal_A &, signal_B &);
};
template <> struct Thread0<Th0ND> {
  constexpr void tick(signal_A &, signal_B &);
};
template <> struct Thread0<E> {
  constexpr void tick(signal_A &, signal_B &);
};

template <> struct Thread1<I> {
  constexpr void tick(signal_A &, signal_B &);
};
template <> struct Thread1<S1> {
  constexpr void tick(signal_A &, signal_B &);
};

template <> struct Thread1<S4> {
  constexpr void tick(signal_A &, signal_B &);
};

template <> struct Thread1<Th1D> {
  constexpr void tick(signal_A &, signal_B &) {}
};
template <> struct Thread1<Th1ND> {
  constexpr void tick(signal_A &, signal_B &) {}
};

template <> struct Thread2<I> {
  constexpr void tick(signal_A &, signal_B &);
};
template <> struct Thread2<S2> {
  constexpr void tick(signal_A &, signal_B &);
};
template <> struct Thread2<S3> {
  constexpr void tick(signal_A &, signal_B &);
};
template <> struct Thread2<Th2ND> {
  constexpr void tick(signal_A &, signal_B &) {}
};
template <> struct Thread2<Th2D> {
  constexpr void tick(signal_A &, signal_B &) {}
};

// XXX: Variables to hold  the state of each thread in the program
static Thread0State st0;
static Thread1State st1;
static Thread2State st2;

// XXX: Initialise all the threads with I
constexpr void init0() { st0 = Thread0<I>{}; }
constexpr void init1() { st1 = Thread1<I>{}; }
constexpr void init2() { st2 = Thread2<I>{}; }

template <class... Ts> struct overloaded : Ts... {
  using Ts::operator()...;
};

constexpr void visit0(Thread0State &ts, signal_A &a, signal_B &b) {
  std::visit(overloaded{[&a, &b](auto &t) { return t.tick(a, b); }}, ts);
}

constexpr void visit1(Thread1State &ts, signal_A &a, signal_B &b) {
  std::visit(overloaded{[&a, &b](auto &t) { return t.tick(a, b); }}, ts);
}
constexpr void visit2(Thread2State &ts, signal_A &a, signal_B &b) {
  std::visit(overloaded{[&a, &b](auto &t) { return t.tick(a, b); }}, ts);
}

// XXX: Now give the code for each thread state
constexpr void Thread0<I>::tick(signal_A &a, signal_B &b) {
  curr_A.status = true;
  st0 = Thread0<S0>{};
}

// XXX: async is just too expensive, in terms of runtime!
constexpr void Thread0<S0>::tick(signal_A &a, signal_B &b) {
  init1();
  init2();
  // XXX: run tick for each thread and gets its state
  // XXX: Make a copy of the signals needed and send them in.
  signal_B bb = b;
  visit1(st1, a, bb);
  // auto f1 = std::async(std::launch::async, [&a, &bb]() { visit1(st1, a, bb);
  // });
  signal_B bbb = b;
  // auto f2 =
  //     std::async(std::launch::async, [&a, &bbb]() { visit2(st2, a, bbb); });
  // f1.get();
  // f2.get();
  visit2(st2, a, bbb);
  // XXX: Here you should combine the values of variables
  if (bb.status)
    b = b + bb;
  if (bbb.status)
    b = b + bbb;
  if ((std::holds_alternative<Thread1<Th1D>>(st1)) and
      (std::holds_alternative<Thread2<Th2D>>(st2))) {
    st0 = Thread0<E>{};
  } else {
    st0 = Thread0<Th0ND>{};
  }
}

constexpr void Thread0<Th0ND>::tick(signal_A &a, signal_B &b) {
  signal_B bb = b;
  visit1(st1, a, bb);
  signal_B bbb = b;
  visit2(st2, a, bbb);
  // XXX: Combine the values
  if (bb.status)
    b = b + bb;
  if (bbb.status)
    b = b + bbb;
  if ((std::holds_alternative<Thread1<Th1D>>(st1)) and
      (std::holds_alternative<Thread2<Th2D>>(st2))) {
    st0 = Thread0<E>{};
  } else {
    st0 = Thread0<Th0ND>{};
  }
}

static bool done = false;
constexpr void Thread0<E>::tick(signal_A &a, signal_B &b) { done = true; }

// XXX: The code for thread1
constexpr void Thread1<I>::tick(signal_A &a, signal_B &b) {
  if (pre_A.status) {
    b.status = true;
    b.value = 10;
    st1 = Thread1<S1>{};
  } else {
    st1 = Thread1<S4>{};
  }
}

constexpr void Thread1<S1>::tick(signal_A &a, signal_B &b) {
  st1 = Thread1<Th1D>{};
}

constexpr void Thread1<S4>::tick(signal_A &a, signal_B &b) {
  st1 = Thread1<Th1D>{};
}

// XXX: The code for thread2
constexpr void Thread2<I>::tick(signal_A &a, signal_B &b) {
  b.status = true;
  b.value = 100;
  st2 = Thread2<S2>{};
}

constexpr void Thread2<S2>::tick(signal_A &a, signal_B &b) {
  st2 = Thread2<S3>{};
}

constexpr void Thread2<S3>::tick(signal_A &a, signal_B &b) {
  st2 = Thread2<Th2D>{};
}

void print_outputs() {
  std::cout << "Status of signal A: " << curr_A.status << "\n";
  std::cout << "Status of signal B: " << curr_B.status << "\n";
  std::cout << "Value of signal B: " << curr_B.value << "\n";
  std::cout << "*****************************************\n";
}

constexpr void reset_signal_status() {
  curr_A.status = false;
  curr_B.status = false;
}

int main(int argc, char *argv[]) {
  // XXX: Now make Thread0 and then make it tick!
  init0();
  while (!done) {
    // XXX: Get input from environment into curr signals
    visit0(st0, curr_A, curr_B);
    // XXX: Now print all the curr signals
    print_outputs();
    // XXX: Now swap curr and pre
    std::swap(pre_A.status, curr_A.status);
    std::swap(pre_B.status, curr_B.status);
    // XXX: Reset the status to 0 for all signals.
    reset_signal_status();
  }
  return 0;
}
