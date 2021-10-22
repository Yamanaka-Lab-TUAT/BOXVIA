# ![Title](./src/assets/title.png "Title") <br>

Bayesian Optimization Executable and Visualizable Application (BOXVIA) is a GUI-based application for Bayesian optimization. By using BOXVIA, users can perform Bayesian optimization and visualize functions obtained from the optimization process (i.e. mean function, its standard deviation, and acquisition function) without construction of a computing environment and programming skills. BOXVIA offers significant help for incorporating Bayesian optimization into your optimization problem.

## Download an executable application
You can download executable files for BOXVIA from [Releases](https://github.com/Yamanaka-Lab-TUAT/BOXVIA/releases).

## How to start BOXVIA
### For using executable file
Extract the downloaded file and double-click on "BOXVIA" executable file. <br>
The name of the executable is "BOXVIA.exe" on windows, "BOXVIA.app" on macOS, and "BOXVIA" on linux.

### For running source code
 Start with
```bash
python main.py
```

## Dependencies 
To start BOXVIA using the source codes, the following libraries are required. <br>
Note: If you use executable file, the dependencies are not necessary. <br>

- GPyOpt
- matplotlib
- pandas
- Dash
- Dash_bootstrap_components


You can install the dependencies by:
```bash
pip install -r requirements.txt
```

## Tutorial
Please see video tutorials uploaded on YouTube. <br>
[Video tutorial for 1D function](https://www.youtube.com/watch?v=ljzGmVSf16U) <br>
[Video tutorial for 5D function](https://www.youtube.com/watch?v=merYNmawvkw)

## License
BSD License (3-clause BSD License)

## Developers' Affiliation
[Yamanaka Research Group @ TUAT](http://web.tuat.ac.jp/~yamanaka/)
