
file(GLOB mobileSSDConsoleSources *.cpp)
file(GLOB mobileSSDConsoleIncludes *.h )

cuda_add_executable(mobileSSD-console ${mobileSSDConsoleSources})

target_link_libraries(mobileSSD-console jetson-inference)

install(TARGETS mobileSSD-console DESTINATION bin)
