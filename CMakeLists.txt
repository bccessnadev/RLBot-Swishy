cmake_minimum_required(VERSION 3.22)

project(RocketForge_Swishy_SwishyV1_v1_03_27B)

set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}")
set(RLBOT ON CACHE BOOL "Build the portable RLBot runtime" FORCE)

set(RLBOT_CPP_BUILD_BENCHMARK OFF CACHE BOOL "Exclude RLBotCPP benchmark" FORCE)

add_subdirectory(ForgeTraining/RLGym)
add_subdirectory(ForgeTraining/GigaLearn/GigaLearnInfer)
add_subdirectory(Bots)
add_subdirectory(RocketForgeRLBot)

target_compile_definitions(RLBot PRIVATE
    RLBOT_DEFAULT_AGENT="Swishy"
)
