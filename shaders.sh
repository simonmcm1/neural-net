glslc -fshader-stage=compute gpu/assets/compute.glsl -o gpu/assets/compute.spv
glslc -fshader-stage=compute gpu/assets/activate.glsl -o gpu/assets/activate.spv
glslc -fshader-stage=compute gpu/assets/reset.glsl -o gpu/assets/reset.spv
glslc -fshader-stage=compute gpu/assets/clear.glsl -o gpu/assets/clear.spv
glslc -fshader-stage=compute gpu/assets/deltas.glsl -o gpu/assets/deltas.spv