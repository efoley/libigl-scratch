include(FetchContent)
FetchContent_Declare(
    h3
    GIT_REPOSITORY https://github.com/uber/h3.git
    GIT_TAG v4.1.0
)
FetchContent_MakeAvailable(h3)
