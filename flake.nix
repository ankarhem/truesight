{
  description = "Rust development environment";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    git-hooks.url = "github:cachix/git-hooks.nix";
    git-hooks.inputs.nixpkgs.follows = "nixpkgs";
    treefmt-nix.url = "github:numtide/treefmt-nix";
    fenix = {
      url = "https://flakehub.com/f/nix-community/fenix/0.1";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs@{ self, ... }:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [
        inputs.git-hooks.flakeModule
        inputs.treefmt-nix.flakeModule
      ];
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
      ];
      perSystem =
        {
          config,
          lib,
          pkgs,
          system,
          ...
        }:
        let
          rustPlatform = pkgs.makeRustPlatform {
            cargo = pkgs.rustToolchain;
            rustc = pkgs.rustToolchain;
          };

          truesight = rustPlatform.buildRustPackage {
            pname = "truesight";
            version = "0.1.0";
            src = ./.;

            cargoLock.lockFile = ./Cargo.lock;

            nativeBuildInputs = with pkgs; [
              pkg-config
              cmake
            ];

            buildInputs = with pkgs; [
              openssl
              onnxruntime
            ];

            doCheck = false;

            postFixup = pkgs.lib.optionalString pkgs.stdenv.hostPlatform.isDarwin ''
              install_name_tool -add_rpath "${pkgs.onnxruntime}/lib" "$out/bin/truesight"
            '';

            LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
            ORT_LIB_PATH = "${pkgs.onnxruntime}/lib";
            ORT_PREFER_DYNAMIC_LINK = "1";
          };
        in
        {
          packages.default = truesight;
          treefmt = {
            programs.nixfmt.enable = true;
            programs.nixfmt.package = pkgs.nixfmt;
            programs.rustfmt.enable = true;
          };
          pre-commit.settings.hooks = {
            treefmt.enable = true;
          };
          devShells.default = pkgs.mkShell {
            inherit (config.pre-commit) shellHook;
            packages =
              with pkgs;
              [
                rustToolchain
                cargo-deny
                cargo-edit
                cargo-watch
                openssl
                pkg-config
                rust-analyzer
              ]
              ++ config.pre-commit.settings.enabledPackages;

            env = {
              RUST_SRC_PATH = "${pkgs.rustToolchain}/lib/rustlib/src/rust/library";
            };
          };
          _module.args.pkgs = import inputs.nixpkgs {
            inherit system;
            overlays = lib.attrValues self.overlays;
          };
        };
      flake.overlays.default = final: prev: {
        nodejs = final.nodejs_24;
        rustToolchain =
          with inputs.fenix.packages.${prev.stdenv.hostPlatform.system};
          combine (
            with stable;
            [
              clippy
              rustc
              cargo
              rustfmt
              rust-src
            ]
          );
      };
    };
}
