# Evolutionary Matrix Factorization (EMF)

This project contains the source code used to evaluate the performance of EMF algorithm.

The source code has been structured has follows:

- **experiments** package contains the main classes to execute all the experiments.
- **mf** package contains the matrix factorization algorithms implementation not included into CF4J library.
- **qualityMeasures** package contains MAE and MSE implementation to evaluate the quality of the predictions performed.

The following experiments has been encoded:

- **GenerateBinaryFiles**. This class splits MovieLens and FilmTrust datasets into training and test sets. Results are stored in a CF4J's binary files. All datasets can be found into *datasets* directory.
- **DatasetInfo**. This class prints the main properties of the datasets.
- **HyperparametersOptimization**. This class performs a grid search to tune the hyper-parameters of the baselines that we are comparing with EMF.
- **GeneticProgrammingOptimization**. This class performs a genetic programming based optimization in order to discover the most accurate function to perform predictions using Matrix Factorization based Collaboartive Filtering.
- **BaselineComparison**. This class compares the functions returned by *GeneticProgrammingOptimization* experiment with several baselines.

The same code has been used to perform experiments within MovieLens and FilmTrust datasets. To switch between both datasets comment/uncomment the corresponding lines.

In *lib* directory contains the *jar* version of the libraries required by this project.