
{ pkgs ? import <nixpkgs> {} }:

pkgs.python3.pkgs.buildPythonPackage rec {
  pname = "hids";
  version = "0.0.1";

  src = ./.;

  propagatedBuildInputs = [
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
  */
    pkgs.python3Packages.numpy
    pkgs.python3Packages.scikit-learn
    pkgs.python3Packages.pytorch
    pkgs.python3Packages.matplotlib
    pkgs.python3Packages.seaborn
    pkgs.tetragon
    pkgs.zlib
  ];

  meta = with pkgs.lib; {
    description = "A Python package to generate secure configuration for systemd service.";
    license = licenses.mit;
    maintainers = with maintainers; [ ];
  };
}

