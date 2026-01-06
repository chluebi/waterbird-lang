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

        rustToolchain = (pkgs.rust-bin.nightly."2025-11-20".default.override {
          extensions = [ "rust-src" "rust-analyzer" "llvm-tools-preview" ];
        });

      in {
        devShells.default = pkgs.mkShell {
          
          nativeBuildInputs = [
            rustToolchain
            pkgs.pkg-config
            pkgs.cmake
            pkgs.cargo-fuzz
          ];

          buildInputs = [
            llvmPackages.llvm
            llvmPackages.clang
            llvmPackages.libclang
            llvmPackages.lld
            pkgs.ncurses
            pkgs.libffi
            pkgs.libxml2
            pkgs.zlib
          ];

          shellHook = ''
            # Critical for bindgen to find libclang.so on NixOS
            export LIBCLANG_PATH="${llvmPackages.libclang.lib}/lib"
            
            # Help nested Cargo builds find libraries
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ llvmPackages.libclang.lib pkgs.zlib pkgs.libffi ]}:$LD_LIBRARY_PATH"

            # Autarkie specific requirements
            export LLVM_DIR="${llvmPackages.llvm}/lib/cmake/llvm"
            export LLVM_CONFIG_PATH="${llvmPackages.llvm}/bin/llvm-config"
            
            # Set compilers
            export CC="${llvmPackages.clang}/bin/clang"
            export CXX="${llvmPackages.clang}/bin/clang++"

            echo "--- NixOS Fuzzing Environment (Autarkie Ready) ---"
            echo "LIBCLANG_PATH: $LIBCLANG_PATH"
            echo "--------------------------------------------------"
          '';
        };
      });
}