# CMakeList.txt : CMake project for ML, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

add_compile_options("/std:c++latest")

project ("ML")

# Add source to this project's executable.
add_executable (ML "ML.cpp" "ML.h" "Network.h" "Network.cpp" "DataPoint.h" "DataPoint.cpp" "networks/mnist.h" "util.cpp" "util.h" "networks/mnist.cpp" "networks/test.h" "networks/test.cpp" "networks/seb.h" "networks/seb.cpp" "Timer.h" "Timer.cpp" "ThreadPool.h" "ThreadPool.cpp" "Logging.h" "Logging.cpp" "CPUTrainer.h" "CPUTrainer.cpp" "gpu/compute.cpp" "gpu/compute.h" "gpu/Buffer.cpp" "gpu/Buffer.h" "gpu/Context.cpp" "gpu/Context.h" "gpu/Pipeline.h" "gpu/Pipeline.cpp" "gpu/GPUNetwork.h")

find_package(Vulkan REQUIRED FATAL_ERROR)
target_link_libraries (ML PRIVATE ${Vulkan_LIBRARY})
target_include_directories (ML
    PUBLIC ${Vulkan_INCLUDE_DIR}
)
#message(STATUS vulkanlib ${Vulkan_LIBRARY})
#message(STATUS vulkanlib ${VULKAN_INCLUDE_DIR})
#message(STATUS vulkanlib ${Vulkan_LIBRARY})

#target_link_directories(ML PRIVATE "${VCPKG_INSTALLED_DIR}/${VCPKG_TAGET_TRIPLET}/debug")
set(Python_ROOT_DIR "%LOCALAPPDATA%/Programs/Python/Python310")
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
target_link_libraries(ML PRIVATE
  Python3::Python
  Python3::Module
)

find_package(nlohmann_json CONFIG REQUIRED)
target_link_libraries(ML PRIVATE nlohmann_json::nlohmann_json)

target_compile_definitions(ML PRIVATE WITHOUT_NUMPY)

find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")
target_include_directories(ML PRIVATE ${MATPLOTLIB_CPP_INCLUDE_DIRS})
target_compile_definitions(ML PRIVATE WITHOUT_NUMPY)


# TODO: Add tests and install targets if needed.
if(WIN32)
    add_custom_command(TARGET ML POST_BUILD 
    COMMAND ${Python3_EXECUTABLE} -m venv $<TARGET_FILE_DIR:ML>/..
    COMMAND $<TARGET_FILE_DIR:ML>/../Scripts/activate.bat
    COMMAND python -m pip install -r ${CMAKE_SOURCE_DIR}/requirements.txt > pip_install.log)
else()
    add_custom_command(TARGET ML POST_BUILD 
    COMMAND ${Python3_EXECUTABLE} -m venv $<TARGET_FILE_DIR:ML>/..
    COMMAND $<TARGET_FILE_DIR:ML>/../Scripts/activate
    COMMAND python -m pip install -r ${CMAKE_SOURCE_DIR}/requirements.txt > pip_install.log)
endif()