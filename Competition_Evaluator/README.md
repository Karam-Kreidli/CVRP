<img src="logo.png" alt="ML4VRP Logo" width="215">

# ML4VRP Competition Resources

This repository is used for the Machine Learning for Evolutionary Computation - for Vehicle Routing Problems ([ML4VRP](https://sites.google.com/view/ml4vrp)) competition in [GECCO 2026](https://gecco-2026.sigevo.org/HomePage). 

In this repository, you will find:
- [Problem Instances](#vrps) for the competition
- - [CVRP](#cvrp)
- - [CVRPTW](#cvrptw)
- [Solution Evaluator](#api)

## <a id='vrps'>VRP Problem Instances </a>
### <a id='cvrp'>CVRP </a>

The X dataset [Uchoa17] is one of the most widely studied CVRP benchmark data sets. This data set covers different instance features, such as depot positioning, customer positioning, demand distribution etc, allowing a comprehensive assessment of algorithm performance. 

The problem instances provided in the competition are the instances in the X dataset with customers ranging from 100 to 400, covering different instance types. The competition will evaluate the submitted solution results using a subset of the provided instances (unknown to the participants before the results are presented).

The CVRP problem data can be found under the `Instances/cvrp/vrp/` directory. The JSON files corresponding to the problem instances can be found under the `Instances/cvrp/json/` directory. 

Each `.vrp` file and json file is named with respect to its corresponding instance name, e.g.: the files corresponding to problem instance **X-n101-k25** is located at
- `Instances/cvrp/vrp/X-n101-k25.vrp` and 
- `Instances/cvrp/json/X-n101-k25.json`. 

Note: The X data set are listed under Uchoa et al. (2014) in CVRPLIB. See the paper [New benchmark instances for the capacitated vehicle routing problem](https://doi.org/10.1016/j.ejor.2016.08.012) for the detailed instance description.

### <a id='cvrptw'>CVRPTW </a>
Solomon [Sol87] dataset and Homberger and Gehring [HG99] data set are widely studied CVRPTW benchmark data sets. Both data sets consist of [six types of instances](http://web.cba.neu.edu/~msolomon/problems.htm), i.e., C1, C2, R1, R2, RC1, RC2, which differ with respect to the customers’ geographical locations, vehicle capacity, density and tightness of the time windows. 

The problem instances provided in the competition are taken from two sources, i.e., 
- Solomon [Sol87] dataset of 100 customer problems,
- Homberger and Gehring [HG99] data sets of 200 customer problems and 400 customer problems.

The provided problem instances provided are randomly selected from these three sized problem instances, covering different instance types. The competition will conduct the evaluation of the submitted solution results using a subset of the provided instances (unknown to the participants before the results are presented). 

The problem instances provided in the competition are available to download on the `Instances/cvrptw/` directory. All the VRPTW instances can also be found in [CVRPLIB](http://vrp.galgos.inf.puc-rio.br/index.php/en/). 

In addition to the benchmark VRPTW instances, we provide an example problem instance `toy`, locating at 
- `Instances/cvrptw/txt/toy.txt` and 
- `Instances/cvrptw/json/toy.json`. 

The text files corresponding to the problem instances can be found under the `Instances/cvrptw/txt/` directory. The JSON files corresponding to the problem instances can be found under the `Instances/cvrptw/json/` directory. 

Each txt file and json file is named with respect to its corresponding instance name, e.g.: the files corresponding to problem instance **C102** is located at
- `Instances/cvrptw/txt/C102.txt` and 
- `Instances/cvrptw/json/C102.json`. 


Note: See [Solomon's website](http://web.cba.neu.edu/~msolomon/problems.htm) for the detailed instance description.

## <a id='api'>VRP Solution Evaluator </a>

The Python script `evaluator.py` is the solution evaluation program to use. The solution evaluator takes a solution and the corresponding problem instance to
- check feasibility of the solution,
- calculate the objective function value of the solution (following the objective function as stated on the [competition website](https://sites.google.com/view/ml4vrp#h.8tn33nmddfdh)) for feasible solution.

<b>Please follow the solution format specified on the [competition website](https://sites.google.com/view/ml4vrp#h.j2mwimqjm1ge).</b>

### How to start

Let's prepare the environment and download the resources to work.
1. Download/clone the whole repository. Note the `Instances/` directory where **JSON file format** of the problem instances are essentially needed.
2. Install `Python 3`.
3. Install the [DEAP](https://github.com/deap/deap) framework in Python.
   ```bash
   pip install deap
   ```
4. Prepare the solution files in the specific format as described in the [competition website](https://sites.google.com/view/ml4vrp#h.j2mwimqjm1ge).


### Usage example
Navigating to the repository directory, use the following command in the terminal or command prompt:
```sh
python evaluator.py <problem_type> <instance_name> <path_to_solution_file>
```
Replace <problem_type>, <instance_name>, and <path_to_solution_file> with appropriate values.
- `<problem_type> `must be one of the following: `cvrp` or `cvrptw`.
- `<instance_name>` is the name of the instance you want to evaluate.
- `<path_to_solution_file>` is the path to the solution file you want to evaluate against.

<div class="alert alert-success" style="display: inline-block;">
For additional examples of usage, please refer to the <a href="./quick-start.ipynb">quick-start</a> notebook.
</div>

## File Structure
```
├── Instances/
│   ├── cvrp/
│   │   ├── json/
│   │   │   ├──<Instance name>.json
│   │   │   └── ...
│   │   ├── vrp/
│   │   │   ├──<Instance name>.vrp
│   │   │   └── ...
│   ├── cvrptw/
│   │   ├── json/
│   │   │   ├──<Instance name>.json
│   │   │   └── ...
│   │   ├── txt/
│   │   │   ├──<Instance name>.txt
│   │   │   └── ...
├── vrp_evaluator/
│   ├── __init__.py
│   ├── core.py
│   └── utils.py
├── evaluator.py
├── Solutions
│   ├── cvrp/
│   │   ├── Instance name of the solution.txt
│   ├── cvrptw/
│   │   ├── toy_solution.txt
│   │   ├── toy_solution_infeasible.txt
├── README.md
└── logo.png
```

## Organisers
Rong Qu,         University of Nottingham, UK, rong.qu@nottingham.ac.uk

Weiyao Meng,     University of Nottingham, UK, weiyao.meng2@nottingham.ac.uk

Isaac Triguero,  University of Granada, Spain, isaaktriguero@go.ugr.es

Mustafa Misir,   Duke Kunshan University, China, mustafa.misir@dukekunshan.edu.cn

## References
[HG99] J. Homberger and H. Gehring, "Two evolutionary metaheuristics for the vehicle routing problem with time windows," INFOR: Information Systems and Operational Research, 37(3):297–318, 1999. [PDF](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=a34e12bf0a30deb56233c26d82a0979987bb6ce4)

[Sol87] M. M. Solomon, "Algorithms for the vehicle routing and scheduling problems with time window constraints," Operations Research, 35(2):254–265, 1987. [PDF](https://www.jstor.org/stable/pdf/170697.pdf?casa_token=ltF2XRa2-nAAAAAA:OV4ClhhdAM_ds_p3-XIzKaz3hDYb9Jy2yHa7-jniGyYLzy2Rg2JC1b-ope2_gtsoQ1eOfFcgeTvtFmGZdPWDACEySwlfASLdRl-mhJRQE4f_6Kc5jJRnYg)

[Uchoa17] Uchoa, E., Pecin, D., Pessoa, A., Poggi, M., Vidal, T., & Subramanian, A. (2017). New benchmark instances for the capacitated vehicle routing problem. European Journal of Operational Research, 257(3), 845-858. [PDF](http://vrp.galgos.inf.puc-rio.br/index.php/en/new-instances)