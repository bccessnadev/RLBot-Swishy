cmake_minimum_required(VERSION 3.8)

project(RLBot)

option(RLBOT_USE_GPU "Use GPU-enabled libtorch for RLBot inference" ON)

add_subdirectory(inc/cpp-interface)

add_executable(RLBot
    src/Runtime/PortableMain.cpp
    src/Runtime/RLBotClient.cpp
    src/Runtime/RLBotPolicyRuntime.cpp
    src/Runtime/RunRLBot.cpp
)
target_include_directories(RLBot PRIVATE src)
target_compile_features(RLBot PRIVATE cxx_std_20)
target_compile_definitions(RLBot PRIVATE
    PROJECT_ROOT="${CMAKE_SOURCE_DIR}"
    RLBOT_USE_GPU=$<IF:$<BOOL:${RLBOT_USE_GPU}>,1,0>
)
target_link_libraries(RLBot PUBLIC GigaLearnInfer Agents RLBotCPP-static)
