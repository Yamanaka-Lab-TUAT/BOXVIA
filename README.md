![Title](./src/assets/title.png "Title") <br>

Bayesian Optimization Executable and Visualizable Application (BOXVIA) is a GUI-based application for Bayesian optimization. By using BOXVIA, users can perform Bayesian optimization and visualize functions obtained from the optimization process (i.e. mean function, its standard deviation, and acquisition function) without construction of a computing environment and programming skills. BOXVIA offers significant help for incorporating Bayesian optimization into your optimization problem.

## Download an executable application
You can download executable files for BOXVIA from [Releases](https://github.com/Yamanaka-Lab-TUAT/BOXVIA/releases).

## Related paper
Akimitsu Ishii, Ryunosuke Kamijyo, Akinori Yamanaka, and Akiyasu Yamamoto, "BOXVIA: Bayesian optimization executable and visualizable application", SoftwareX, Vol. 18 (2022/6), 101019. [Link to ScienceDirect](https://doi.org/10.1016/j.softx.2022.101019) (Open Access Paper)

## How to start BOXVIA
### For using executable file
Extract the downloaded file and double-click on "BOXVIA" executable file. <br>
The name of the executable is "BOXVIA.exe" on windows, "BOXVIA.app" on macOS, and "BOXVIA" on linux.

### For running source code
**Notification!** <br>
We strongly recommend running BOXVIA via the executable file <br>
because the application has not yet support the latest versions of dependencies. <br>
If you run BOXVIA with its source code, carefully check the version of the dependencies.

 Start with
```bash
python main.py
```

## Dependencies 
To start BOXVIA via the source codes, the following libraries are required. <br>
**If you use executable file, no dependencies are required.** <br>

- GPyOpt
- matplotlib
- pandas
- Dash == 2.0.0
- Dash_bootstrap_components == 1.0.0


You can install the dependencies by:
```bash
pip install -r requirements.txt
```
<br>
Note: <br>

If you start BOXVIA via the source codes, the following two codes of
[GPyOpt](https://github.com/SheffieldML/GPyOpt) need to be modified. <br>

- GPyOpt/core/bo.py
- GPyOpt/core/evaluators/batch_local_penalization.py

Please replace these codes with the codes contained in src/GPyOpt_modified in this repository.


## Tutorial
Please see video tutorials uploaded on YouTube. <br>

Video tutorial for 1D function <br>
[![Video tutorial 1D](https://user-images.githubusercontent.com/92300126/173714183-350ed39b-7d02-431b-9260-9679fda73da8.jpg)](https://www.youtube.com/watch?v=ljzGmVSf16U)

Video tutorial for 5D function <br>
[![Video tutorial 5D](https://user-images.githubusercontent.com/92300126/173714468-2f804c6c-7aa6-49c0-9141-fc5878f1ea10.jpg)](https://www.youtube.com/watch?v=merYNmawvkw) 

Further detailes are described in our [paper](https://doi.org/10.1016/j.softx.2022.101019). <br>
Text-based tutorial has been posted on [Qiita](https://qiita.com/akmt-ishii/items/1d5354a1f1f75556281a) (in Japanese).

## License
BSD License (3-clause BSD License)

## Citation
```bash
@article{ISHII2022101019,
title = {BOXVIA: Bayesian optimization executable and visualizable application},
journal = {SoftwareX},
volume = {18},
pages = {101019},
year = {2022},
issn = {2352-7110},
doi = {https://doi.org/10.1016/j.softx.2022.101019},
url = {https://www.sciencedirect.com/science/article/pii/S2352711022000243},
author = {Akimitsu Ishii and Ryunosuke Kamijyo and Akinori Yamanaka and Akiyasu Yamamoto},
}
```

## Developers' Affiliation
[Yamanaka Research Group @ TUAT](http://web.tuat.ac.jp/~yamanaka/)
