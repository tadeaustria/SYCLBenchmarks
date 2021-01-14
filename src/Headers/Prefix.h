#pragma once

#include <cstdlib>
#include <cmath>
#include <algorithm>

bool floatCompare(float f1, float f2)
{
   return std::abs(f1 - f2) <= 1.0e-4f * std::max(std::abs(f1), std::abs(f2));
}