cmake_minimum_required(VERSION 3.16)
project(mooncake-pg)

include_directories(include)
add_subdirectory(src)

if(BUILD_UNIT_TESTS)
  add_executable(pg_nccl_failure_control_test
                 tests/nccl_failure_control_test.cpp)
  target_compile_features(pg_nccl_failure_control_test PRIVATE cxx_std_20)
  target_link_libraries(
    pg_nccl_failure_control_test
    PRIVATE mooncake_pg yalantinglibs::yalantinglibs asio_shared gtest
            gtest_main)
  add_test(NAME pg_nccl_failure_control_test
           COMMAND pg_nccl_failure_control_test)
endif()
