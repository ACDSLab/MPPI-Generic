/*
 * Created on Mon Jun 01 2020 by Bogdan
 *
 */
#ifndef MATH_UTILS_H_
#define MATH_UTILS_H_

// Needed for sampling without replacement
#include <unordered_set>
#include <random>
#include <vector>
#include <algorithm>

// Based off of https://gormanalysis.com/blog/random-numbers-in-cpp
std::vector<int> sample_without_replacement(int k, int N,
    std::default_random_engine g = std::default_random_engine()) {
  if (k > N) {
    throw std::logic_error("Can't sample more than n times without replacement");
  }
  // Create an unordered set to store the samples
  std::unordered_set<int> samples;

  // For loop runs k times
  for (int r = N - k; r < N; r++) {
    int v = std::uniform_int_distribution<>(1, r)(g); // sample between 1 and r
    if (!samples.insert(v).second) { // if v exists in the set
      samples.insert(r);
    }
  }
  // Copy set into a vector
  std::vector<int> final_sequence(samples.begin(), samples.end());
  // Shuffle the vector to get the final sequence of sampling
  std::shuffle(final_sequence.begin(), final_sequence.end(), g);
  return final_sequence;
}

#endif
