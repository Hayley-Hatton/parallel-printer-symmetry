# Meta
cmake_minimum_required(VERSION 3.15)
project(
	parallelprint
	DESCRIPTION "Parallel Printer"
	HOMEPAGE_URL "http://www.hayleyhatton.co.uk")

# C++ version control
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Dependencies
find_package(CGAL)
find_package(OpenMP)

# Compile
add_executable(parallelprint ParallelPrinter.h ParallelPrinter.cpp)
target_link_libraries(parallelprint CGAL::CGAL OpenMP::OpenMP_CXX)



