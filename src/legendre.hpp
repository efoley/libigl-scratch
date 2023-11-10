#pragma once

#include <cmath>
#include <concepts>
#include <iostream>

// Function to calculate Legendre polynomial of degree n at x
double legendrePolynomial(int n, double x) {
  if (n == 0)
    return 1.0; // P_0(x) = 1
  else if (n == 1)
    return x; // P_1(x) = x

  double P_n_minus_two = 1.0; // P_0(x)
  double P_n_minus_one = x;   // P_1(x)
  double P_n = 0.0;

  for (int k = 2; k <= n; ++k) {
    P_n = ((2 * k - 1) * x * P_n_minus_one - (k - 1) * P_n_minus_two) / k;
    P_n_minus_two = P_n_minus_one;
    P_n_minus_one = P_n;
  }

  return P_n;
}

int factorial(int n) { return (n == 0) ? 1 : n * factorial(n - 1); }

int doubleFactorial(int n) {
  int result = 1;
  for (int i = n; i > 0; i -= 2) {
    result *= i;
  }
  return result;
}

// Function to calculate the associated Legendre polynomial P_l^m(x)
double associatedLegendrePolynomial(int l, int m, double x) {
  if (m < 0) {
    m = -m; // Take the absolute value of m
    // Use the relation between P_l^-m(x) and P_l^m(x) here
    double result = associatedLegendrePolynomial(l, m, x);
    auto phase = m % 2 == 0 ? 1 : -1; // Include the phase factor (-1)^m
    result *= phase * factorial(l - m) / factorial(l + m);
    return result;
  }

  // Check if m > l, which would be zero
  if (m > l)
    return 0.0;

  // Starting values for P_m^m and P_{m+1}^m
  double pmm = std::pow(-1, m) * doubleFactorial(2 * m - 1) *
               std::pow(1 - x * x, m / 2.0);
  if (l == m)
    return pmm;

  double pmmp1 = x * (2 * m + 1) * pmm;
  if (l == m + 1)
    return pmmp1;

  double pll = 0.0;
  for (int ll = m + 2; ll <= l; ++ll) {
    pll = (x * (2 * ll - 1) * pmmp1 - (ll + m - 1) * pmm) / (ll - m);
    pmm = pmmp1;
    pmmp1 = pll;
  }

  return pll;
}