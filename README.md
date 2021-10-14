![Title](./src/assets/title.png "Title")
# 
Bayesian Optimization Executable and Visualizable Application (BOXVIA) is a GUI-based application for Bayesian optimization. BOXVIA enables to perform Bayesian optimization and visualize distributions of functions obtained from the optimization process (i.e. mean function, its standard deviation, and acquisition function) without any coding. You will find BOXVIA useful in efficiently solving a variety of optimization problems, such as design of your experiments.

## Download as executable application
You can download executable files for BOXVIA from [Releases](https://github.com/Yamanaka-Lab-TUAT/BOXVIA/releases).

## How to start BOXVIA
### For using executable file
Extract the downloaded file and  double-click on "BOXVIA" executable file. <br>
The name of the executable is "BOXVIA.exe" on windows, "BOXVIA.app" on macOS, and simply "BOXVIA" for linux.

### For running source code
 Start with
```bash
python main.py
```

## Dependencies 
To start BOXVIA using the source codes, the following libraries are required. <br>
Note: If you use executable file, the dependencies are not necessary.) <br>

- GPyOpt
- matplotlib
- pandas
- Dash
- Dash_bootstrap_components

and these libraries' dependencies (e.g. numpy, scipy, etc, but those will usually be installed together if the above libraries are installed via pip.)


You can install the dependencies by:
```bash
pip install -r requirements.txt
```

## Usage
Please refer to our Journal published in the future.

## License
BSD License (3-clause BSD License)

## Developers' Affiliation
[Yamanaka Research Group @ TUAT](http://web.tuat.ac.jp/~yamanaka/)
