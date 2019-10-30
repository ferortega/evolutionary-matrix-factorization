package main.java.mf;

import cf4j.*;
import cf4j.model.matrixFactorization.FactorizationModel;
import sym_derivation.symderivation.SymFunction;

import java.util.HashMap;

public class Emf implements FactorizationModel {

    private double learningRate;
    private double regularization;
    private int numFactors;
    private int numIters;

    private SymFunction sf;

    private double [][] p;
    private double [][] q;

    public Emf (String func, int numFactors, int numIters, double regularization, double learningRate) {

        // create model function
        this.sf = SymFunction.parse(func);

        // model hyper-parameters
        this.numFactors = numFactors;
        this.numIters = numIters;
        this.regularization = regularization;
        this.learningRate = learningRate;

        // users factors initialization
        this.p = new double [Kernel.getInstance().getNumberOfUsers()][numFactors];
        for (int u = 0; u < Kernel.gi().getNumberOfUsers(); u++) {
            this.p[u] = this.random(this.numFactors, 0, 1);
        }

        // items factors initialization
        this.q = new double [Kernel.getInstance().getNumberOfItems()][numFactors];
        for (int i = 0; i < Kernel.gi().getNumberOfItems(); i++) {
            this.q[i] = this.random(this.numFactors, 0, 1);
        }
    }

    public void train () {

        System.out.println("\nProcessing EMF...");

        // partial derivatives of the model function

        SymFunction [] puSfDiff = new SymFunction [this.numFactors];
        SymFunction [] qiSfDiff = new SymFunction [this.numFactors];

        for (int k = 0; k < this.numFactors; k++) {
            puSfDiff[k] = sf.diff("pu" + k);
            qiSfDiff[k] = sf.diff("qi" + k);
        }

        // repeat numIters times
        for (int iter = 1; iter <= this.numIters; iter++) {

            // compute gradient
            double[][] dp = new double[Kernel.getInstance().getNumberOfUsers()][this.numFactors];
            double[][] dq = new double[Kernel.getInstance().getNumberOfItems()][this.numFactors];

            for (int userIndex = 0; userIndex < Kernel.getInstance().getNumberOfUsers(); userIndex++) {

                User user = Kernel.gi().getUserByIndex(userIndex);

                int itemIndex = 0;

                for (int i = 0; i < user.getNumberOfRatings(); i++) {

                    while (Kernel.gi().getItemByIndex(itemIndex).getItemCode() < user.getItemAt(i)) itemIndex++;

                    HashMap <String, Double> params = getParams(p[userIndex], q[itemIndex]);

                    double prediction = sf.eval(params);
                    double error = user.getRatingAt(i) - prediction;

                    for (int k = 0; k < this.numFactors; k++) {
                        dp[userIndex][k] += this.learningRate * (error * puSfDiff[k].eval(params) - this.regularization * p[userIndex][k]);
                        dq[itemIndex][k] += this.learningRate * (error * qiSfDiff[k].eval(params) - this.regularization * q[itemIndex][k]);
                    }
                }
            }

            // update users factors
            for (int userIndex = 0; userIndex < Kernel.getInstance().getNumberOfUsers(); userIndex++) {
                for (int k = 0; k < this.numFactors; k++) {
                    p[userIndex][k] += dp[userIndex][k];
                }
            }

            // update items factors
            for (int itemIndex = 0; itemIndex < Kernel.getInstance().getNumberOfItems(); itemIndex++) {
                for (int k = 0; k < this.numFactors; k++) {
                    q[itemIndex][k] += dq[itemIndex][k];
                }
            }

            if ((iter % 10) == 0) System.out.print(".");
            if ((iter % 100) == 0) System.out.println(iter + " iterations");
        }
    }

    public double getPrediction (int userIndex, int itemIndex) {
        HashMap <String, Double> params = getParams(this.p[userIndex], this.q[itemIndex]);
        return sf.eval(params);
    }

    private double random (double min, double max) {
        return Math.random() * (max - min) + min;
    }

    private double [] random (int size, double min, double max) {
        double [] d = new double [size];
        for (int i = 0; i < size; i++) d[i] = this.random(min, max);
        return d;
    }

    private HashMap<String, Double> getParams (double [] pu, double [] qi) {
        HashMap <String, Double> map = new HashMap<>();
        for (int k = 0; k < this.numFactors; k++) {
            map.put("pu" + k, pu[k]);
            map.put("qi" + k, qi[k]);
        }
        return map;
    }
}
