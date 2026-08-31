uv pip uninstall \
  quack-kernels \
  nvidia-cutlass-dsl \
  nvidia-cutlass-dsl-libs-base \
  nvidia-cutlass-dsl-libs-core \
  nvidia-cutlass-dsl-libs-cu12 \
  nvidia-cutlass-dsl-libs-cu13

uv pip install --no-cache --no-deps \
  "nvidia-cutlass-dsl-libs-base==4.5.2" \
  "nvidia-cutlass-dsl-libs-cu13==4.5.2" \
  "nvidia-cutlass-dsl==4.5.2" \
  "quack-kernels==0.5.0"