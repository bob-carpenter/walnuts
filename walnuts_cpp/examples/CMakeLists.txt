

add_executable(test_nuts test.cpp)
target_link_libraries(test_nuts PRIVATE Eigen3::Eigen walnuts::walnuts)
