FROM ubuntu:24.04

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# Bob builds both native Linux and Windows packages from the same source tree.
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        gnupg \
        lsb-release \
        make \
        ninja-build \
        pkg-config \
        software-properties-common \
        unzip \
        wget \
    && rm -rf /var/lib/apt/lists/*

# Clang 19 can consume the current Windows CRT headers downloaded by xwin.
RUN wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key \
        | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc >/dev/null && \
    add-apt-repository -y \
        "deb https://apt.llvm.org/$(lsb_release -sc)/ llvm-toolchain-$(lsb_release -sc)-19 main" && \
    apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        clang-19 \
        clang-tidy-19 \
        libc++-19-dev \
        libc++abi-19-dev \
        lld-19 \
        llvm-19-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-19 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-19 100 && \
    update-alternatives --install /usr/bin/lld lld /usr/bin/lld-19 100 && \
    update-alternatives --install /usr/bin/ld.lld ld.lld /usr/bin/ld.lld-19 100 && \
    update-alternatives --install /usr/bin/llvm-ar llvm-ar /usr/bin/llvm-ar-19 100 && \
    update-alternatives --install /usr/bin/llvm-ranlib llvm-ranlib /usr/bin/llvm-ranlib-19 100 && \
    update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-19 100 && \
    update-alternatives --install /usr/bin/lld-link lld-link /usr/bin/lld-link-19 100 && \
    update-alternatives --install /usr/bin/llvm-dlltool llvm-dlltool /usr/bin/llvm-dlltool-19 100 && \
    update-alternatives --install /usr/bin/llvm-lib llvm-lib /usr/bin/llvm-lib-19 100

# xwin supplies the redistributable Windows CRT and SDK without requiring MSVC.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path
ENV PATH="/root/.cargo/bin:$PATH"
RUN cargo install xwin --locked
RUN xwin --accept-license splat --output /tmp/xwin

# Match the CPU LibTorch release used by the Bob-compatible GGLBot build.
RUN curl -fsSL -o /tmp/libtorch-linux.zip \
        "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcpu.zip" && \
    unzip -q /tmp/libtorch-linux.zip -d /tmp && \
    mv /tmp/libtorch /tmp/libtorch-linux && \
    rm -f /tmp/libtorch-linux.zip

RUN curl -fsSL -o /tmp/libtorch-win.zip \
        "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.7.0%2Bcpu.zip" && \
    unzip -q /tmp/libtorch-win.zip -d /tmp && \
    mv /tmp/libtorch /tmp/libtorch-win && \
    rm -f /tmp/libtorch-win.zip

WORKDIR /src
COPY . /src

# clang-cl wrappers keep CMake and LibTorch on their MSVC-compatible code paths.
RUN printf '#!/bin/bash\nexec /usr/bin/clang++ --driver-mode=cl -target x86_64-pc-windows-msvc "$@"\n' \
        > /usr/local/bin/clang-cl && \
    printf '#!/bin/bash\nexec /usr/bin/clang --driver-mode=cl -target x86_64-pc-windows-msvc "$@"\n' \
        > /usr/local/bin/clang-cl-c && \
    chmod +x /usr/local/bin/clang-cl /usr/local/bin/clang-cl-c

RUN mkdir -p /src/cmake && \
    cat > /src/cmake/toolchain-msvc.cmake <<'TOOLCHAIN_EOF'
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_VERSION 10.0)

set(CMAKE_C_COMPILER /usr/local/bin/clang-cl-c)
set(CMAKE_CXX_COMPILER /usr/local/bin/clang-cl)
set(CMAKE_C_COMPILER_TARGET x86_64-pc-windows-msvc)
set(CMAKE_CXX_COMPILER_TARGET x86_64-pc-windows-msvc)

set(CMAKE_LINKER lld-link)
set(CMAKE_AR llvm-lib)
set(CMAKE_RANLIB llvm-lib)
set(CMAKE_C_CREATE_STATIC_LIBRARY "<CMAKE_AR> /OUT:<TARGET> <OBJECTS>")
set(CMAKE_CXX_CREATE_STATIC_LIBRARY "<CMAKE_AR> /OUT:<TARGET> <OBJECTS>")

set(CMAKE_C_FLAGS_INIT
  "-MD /imsvc/tmp/xwin/crt/include /imsvc/tmp/xwin/sdk/include/ucrt /imsvc/tmp/xwin/sdk/include/um /imsvc/tmp/xwin/sdk/include/shared /clang:-Wno-unused-command-line-argument"
)
set(CMAKE_CXX_FLAGS_INIT
  "-MD -Xclang -stdlib=libc++ /imsvc/tmp/xwin/crt/include /imsvc/tmp/xwin/sdk/include/ucrt /imsvc/tmp/xwin/sdk/include/um /imsvc/tmp/xwin/sdk/include/shared /clang:-Wno-unused-command-line-argument"
)
set(CMAKE_EXE_LINKER_FLAGS_INIT
  "/LIBPATH:/tmp/xwin/crt/lib/x86_64 /LIBPATH:/tmp/xwin/sdk/lib/um/x86_64 /LIBPATH:/tmp/xwin/sdk/lib/ucrt/x86_64"
)

set(CMAKE_TRY_COMPILE_CONFIGURATION Release)
set(CMAKE_FIND_LIBRARY_SUFFIXES .lib .dll .dll.a .a)
set(CMAKE_FIND_ROOT_PATH /tmp/xwin/crt /tmp/xwin/sdk /tmp/libtorch-win)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
TOOLCHAIN_EOF

# Build a CPU-only runtime; the packaged bot therefore works without CUDA.
RUN cmake -S . -B build-linux -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLIBTORCH_ROOT=/tmp/libtorch-linux \
        -DRLBOT_USE_GPU=OFF && \
    cmake --build build-linux --target RLBot --parallel "$(nproc)"

# RLBotCPP enables one MSVC-only preprocessor flag which clang-cl does not need.
RUN sed -i '/Zc:preprocessor/d' /src/RocketForgeRLBot/inc/cpp-interface/library/CMakeLists.txt
RUN cmake -S . -B build-win -G Ninja \
        -DCMAKE_TOOLCHAIN_FILE=/src/cmake/toolchain-msvc.cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DLIBTORCH_ROOT=/tmp/libtorch-win \
        -DRLBOT_USE_GPU=OFF && \
    cmake --build build-win --target RLBot --parallel "$(nproc)"

RUN mkdir -p \
        /out/x86_64-unknown-linux-gnu \
        /out/x86_64-pc-windows-msvc

RUN test -f /src/build-linux/RLBot && \
    cp /src/build-linux/RLBot /out/x86_64-unknown-linux-gnu/RLBot
RUN test -f /src/build-win/RLBot.exe && \
    cp /src/build-win/RLBot.exe /out/x86_64-pc-windows-msvc/RLBot.exe

# Copy the minimal native LibTorch closure needed for Linux inference.
RUN cp /tmp/libtorch-linux/lib/libtorch_cpu.so /out/x86_64-unknown-linux-gnu/ && \
    cp /tmp/libtorch-linux/lib/libc10.so /out/x86_64-unknown-linux-gnu/ && \
    cp /tmp/libtorch-linux/lib/libtorch.so /out/x86_64-unknown-linux-gnu/ && \
    cp /tmp/libtorch-linux/lib/libtorch_global_deps.so /out/x86_64-unknown-linux-gnu/ && \
    (cp /tmp/libtorch-linux/lib/libgomp*.so* /out/x86_64-unknown-linux-gnu/ 2>/dev/null || true)

# Follow PE imports so the Windows package contains only the required Torch DLLs.
RUN cat > /tmp/copy-pe-dll-closure.sh <<'DLL_EOF'
#!/usr/bin/env bash
set -euo pipefail

out_dir="$1"
search_dir="$2"
shift 2

objdump="$(command -v llvm-objdump-19 || command -v llvm-objdump || command -v objdump)"
queue_file=/tmp/pe-dll-queue.txt
scanned_file=/tmp/pe-dll-scanned.txt
: > "$queue_file"
: > "$scanned_file"

for seed in "$@"; do
    if [ -f "$seed" ]; then
        printf '%s\n' "$seed" >> "$queue_file"
    else
        source_file="$(find "$search_dir" -maxdepth 1 -type f -iname "$seed" -print -quit)"
        if [ -n "$source_file" ]; then
            cp -n "$source_file" "$out_dir/"
            printf '%s\n' "$out_dir/$(basename "$source_file")" >> "$queue_file"
        fi
    fi
done

while [ -s "$queue_file" ]; do
    binary="$(sed -n '1p' "$queue_file")"
    tail -n +2 "$queue_file" > "$queue_file.next" || true
    mv "$queue_file.next" "$queue_file"
    [ -f "$binary" ] || continue

    key="$(basename "$binary" | tr '[:upper:]' '[:lower:]')"
    if grep -Fxq "$key" "$scanned_file"; then
        continue
    fi
    printf '%s\n' "$key" >> "$scanned_file"

    while IFS= read -r dependency; do
        dependency="$(printf '%s' "$dependency" | tr -d '\r')"
        dependency_key="$(printf '%s' "$dependency" | tr '[:upper:]' '[:lower:]')"
        if grep -Fxq "$dependency_key" "$scanned_file"; then
            continue
        fi

        source_file="$(find "$search_dir" -maxdepth 1 -type f -iname "$dependency" -print -quit)"
        if [ -n "$source_file" ]; then
            cp -n "$source_file" "$out_dir/"
            printf '%s\n' "$out_dir/$(basename "$source_file")" >> "$queue_file"
        fi
    done < <("$objdump" -p "$binary" | sed -n 's/^[[:space:]]*DLL Name: //p')
done
DLL_EOF

RUN sed -i 's/\r$//' /tmp/copy-pe-dll-closure.sh && \
    chmod +x /tmp/copy-pe-dll-closure.sh && \
    /tmp/copy-pe-dll-closure.sh \
        /out/x86_64-pc-windows-msvc \
        /tmp/libtorch-win/lib \
        /out/x86_64-pc-windows-msvc/RLBot.exe \
        torch.dll \
        torch_cpu.dll \
        torch_global_deps.dll \
        c10.dll

# Models and portable runtime data live beside each platform executable.
RUN for platform in x86_64-unknown-linux-gnu x86_64-pc-windows-msvc; do \
        cp /src/rlbot/POLICY.lt "/out/$platform/POLICY.lt"; \
        if [ -f /src/rlbot/SHARED_HEAD.lt ]; then \
            cp /src/rlbot/SHARED_HEAD.lt "/out/$platform/SHARED_HEAD.lt"; \
        fi; \
        if [ -f /src/rlbot/agent_setup.json ]; then \
            cp /src/rlbot/agent_setup.json "/out/$platform/agent_setup.json"; \
        fi; \
        cp -a /src/ForgeTraining/collision_meshes "/out/$platform/collision_meshes"; \
    done

ENTRYPOINT ["tar", "-C", "/out", "-cf", "-", \
    "x86_64-unknown-linux-gnu", \
    "x86_64-pc-windows-msvc"]
