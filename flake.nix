{
  description = "Combined Rust and LLVM C++ project with Nix";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = inputs@{ self, nixpkgs, flake-utils, rust-overlay, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };
        
        llvmPackages = pkgs.llvmPackages_18;

        rustToolchain = pkgs.rust-bin.selectLatestNightlyWith (toolchain: toolchain.default.override {
          extensions = [ "rust-src" "rust-analyzer" ];
        });

      in {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            llvmPackages.llvm
            llvmPackages.clang
            llvmPackages.clang-unwrapped
            llvmPackages.libclang
            llvmPackages.lld
            pkgs.cmake
            pkgs.gdb
            pkgs.ncurses
            pkgs.libffi
            pkgs.libxml2
            
            rustToolchain
            pkgs.cargo-fuzz
            pkgs.rustfmt
            pkgs.linuxKernel.packages.linux_zen.perf
          ];

          shellHook = ''
            export LLVM_DIR=${llvmPackages.llvm}/lib/cmake/llvm
            export CLANG_DIR=${llvmPackages.clang}/lib/cmake/clang
            export CXX=${llvmPackages.clang}/bin/clang++
            export CC=${llvmPackages.clang}/bin/clang
            export LLVM_INCLUDE_DIR="${llvmPackages.llvm.dev}/include"
            
            # Ensure cargo-fuzz knows we are on nightly
            export RUSTUP_TOOLCHAIN=nightly 

            echo "--- LLVM C++ & Rust Dev Shell (Nightly) ---"
            echo "Rust version: $(rustc --version)"
            echo "LLVM_DIR: $LLVM_DIR"
            echo "-------------------------------------------"
          '';
        };
      });
}