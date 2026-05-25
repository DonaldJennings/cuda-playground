add_executable(range-profile-cpu
    ${CMAKE_CURRENT_SOURCE_DIR}/range-profile.cpp
)

find_package(PkgConfig REQUIRED)

pkg_check_modules(FFTW REQUIRED IMPORTED_TARGET fftw3f)

target_link_libraries(range-profile-cpu PRIVATE PkgConfig::FFTW)