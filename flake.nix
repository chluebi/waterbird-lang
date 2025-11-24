{
  description = "Combined Rust and LLVM C++ project with Nix";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = inputs@{ self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        llvmPackages = pkgs.llvmPackages_18;

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
            
            pkgs.rustc
            pkgs.cargo
            pkgs.rustfmt
            pkgs.rust-analyzer
            pkgs.linuxKernel.packages.linux_zen.perf
          ];

          shellHook = ''
            export LLVM_DIR=${llvmPackages.llvm}/lib/cmake/llvm
            export CLANG_DIR=${llvmPackages.clang}/lib/cmake/clang
            export CXX=${llvmPackages.clang}/bin/clang++
            export CC=${llvmPackages.clang}/bin/clang
            export LLVM_INCLUDE_DIR="${llvmPackages.llvm.dev}/include"
            echo "--- LLVM C++ & Rust Dev Shell ---"
            echo "LLVM_DIR: $LLVM_DIR"
            echo "CXX: $CXX"
            echo "-------------------------------------"
          '';
        };
      });
}
