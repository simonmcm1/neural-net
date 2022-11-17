#!/bin/bash
export CC=gcc
export CXX=g++-11

mkdir -p build
#rm -rf build/*
cd build

cmake -G Ninja -DCMAKE_TOOLCHAIN_FILE=/opt/vcpkg/scripts/buildsystems/vcpkg.cmake ..
ninja
