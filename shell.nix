{ pkgs ? import <nixpkgs> {} }:

let
  pythonPackages = pkgs.python311Packages; 
in
pkgs.mkShell {
  buildInputs = [
  /*
    (pkgs.python311Packages.joblib.overridePythonAttrs (old: { version = "1.4.0"; })) 
    (pkgs.python311Packages.matplotlib.overridePythonAttrs (old: { version = "3.8.4"; })) 
    (pkgs.python311Packages.networkx.overridePythonAttrs (old: { version = "3.3"; })) 
    (pkgs.python311Packages.numpy.overridePythonAttrs (old: { version = "1.26.4"; })) 
    (pkgs.python311Packages.pandas.overridePythonAttrs (old: { version = "2.2.1"; })) 
    (pkgs.python311Packages.psutil.overridePythonAttrs (old: { version = "5.9.8"; })) 
    (pkgs.python311Packages.scikit-learn.overridePythonAttrs (old: { version = "0.24.2"; })) 
    (pkgs.python311Packages.seaborn.overridePythonAttrs (old: { version = "0.13.2"; })) 
    (pkgs.python311Packages.pytorch.overridePythonAttrs (old: { version = "2.1.2"; }))
    pkgs.python311Full 
    pkgs.python311Packages.pip
    pkgs.python311Packages.virtualenv
    pkgs.python311Packages.setuptools
    pkgs.python311Packages.wheel

    pkgs.tetragon
    pkgs.zlib
 */
    pkgs.gcc
    pkgs.python311Full
    pkgs.python311Packages.virtualenv

  ];
/*
  shellHook = ''
      export PYTHONPATH=$PWD:$PYTHONPATH
      export LD_LIBRARY_PATH=${pkgs.zlib}/li:$LD_LIBRARY_PATH
    export PYTHONPATH="$PWD:$PYTHONPATH"
    export HIDS_BASE_DIR="$PWD/data"
    export HIDS_LOG_DIR="$PWD/logs"
    export HIDS_CONFIG_FILE="$PWD/nix/config.yaml"
    export TETRAGON_BPF_LIB="${pkgs.tetragon}/lib/tetragon/bpf/"
    echo "Welcome to your Python development environment."
  '';
  */
  shellHook = ''
    if [ ! -d .venv ]; then
      virtualenv .venv
      source .venv/bin/activate
    else
      source .venv/bin/activate
      #pip install sh
    fi
    echo "Welcome to your Python development environment."
  '';

}
