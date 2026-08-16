cmake_minimum_required(VERSION 3.8)

project(RLGym)

add_library(RLGym STATIC
    src/RLGymCPP/Gamestates/GameState.cpp
    src/RLGymCPP/Gamestates/Player.cpp
    src/RLGymCPP/Gamestates/StateUtil.cpp
    src/RLGymCPP/ActionParsers/DefaultAction.cpp
)
target_include_directories(RLGym PUBLIC src/)
target_compile_features(RLGym PUBLIC cxx_std_20)

add_subdirectory(RocketSim)
target_link_libraries(RLGym PUBLIC RocketSim)
