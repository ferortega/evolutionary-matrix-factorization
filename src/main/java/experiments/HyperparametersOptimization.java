package main.java.experiments;

import cf4j.Kernel;
import cf4j.Processor;
import cf4j.model.matrixFactorization.Bmf;
import cf4j.model.matrixFactorization.Pmf;
import cf4j.model.predictions.FactorizationPrediction;
import cf4j.qualityMeasures.MAE;
import cf4j.utils.Range;
import main.java.mf.Nmf;
import main.java.qualityMeasures.QualityMeasures;

public class HyperparametersOptimization {

    private static final String BINARY_FILE = "datasets/ml100k.cf4j";
//    private static final String BINARY_FILE = "datasets/filmtrust.cf4j";

    private static int NUM_ITERS = 100;


    public static void main(String[] args) {

        Kernel.getInstance().readKernel(BINARY_FILE);


        // Test PMF model

        double minError = Double.MAX_VALUE;
        String best = "";

        for (int numTopics : Range.ofIntegers(4,2,5)) {
            for (double lambda : Range.ofDoubles(0.005, 0.005, 20)) {
                for (double gamma : Range.ofDoubles(0.005, 0.005, 20)) {

                    Pmf pmf = new Pmf(numTopics, NUM_ITERS, lambda, gamma, false);
                    pmf.train();

                    double mae = QualityMeasures.MAE(pmf);

                    System.out.println("numTopics = " + numTopics + "; lambda = " + lambda + "; gamma = " + gamma + "; mae = " + mae);

                    if (mae < minError) {
                        minError = mae;
                        best = "numTopics = " + numTopics + "; lambda = " + lambda + "; gamma = " + gamma + "; mae = " + mae;
                    }
                }
            }
        }

        System.out.println("\nBest result for PMF => " + best);


        // Test BiasedMF model

        minError = Double.MAX_VALUE;
        best = "";

        for (int numTopics : Range.ofIntegers(4,2,5)) {
            for (double lambda : Range.ofDoubles(0.005, 0.005, 20)) {
                for (double gamma : Range.ofDoubles(0.005, 0.005, 20)) {

                    Pmf biasedMF = new Pmf(numTopics, NUM_ITERS, lambda, gamma);
                    biasedMF.train();

                    double mae = QualityMeasures.MAE(biasedMF);

                    System.out.println("numTopics = " + numTopics + "; lambda = " + lambda + "; gamma = " + gamma + "; mae = " + mae);

                    if (mae < minError) {
                        minError = mae;
                        best = "numTopics = " + numTopics + "; lambda = " + lambda + "; gamma = " + gamma + "; mae = " + mae;
                    }
                }
            }
        }

        System.out.println("\nBest result for BiasedMF => " + best);


        // Test NMF model

        minError = Double.MAX_VALUE;
        best = "";

        for (int numTopics : Range.ofIntegers(4,2,5)) {

            Nmf nmf = new Nmf(numTopics, NUM_ITERS);
            nmf.train();

            double mae = QualityMeasures.MAE(nmf);

            System.out.println("numTopics = " + numTopics + "; mae = " + mae);

            if (mae < minError) {
                minError = mae;
                best = "numTopics = " + numTopics + "; mae = " + mae;
            }
        }

        System.out.println("\nBest result for NMF => " + best);


        // Test BNMF model

        minError = Double.MAX_VALUE;
        best = "";

        for (int numTopics : Range.ofIntegers(4,2,5)) {
            for (double alpha : Range.ofDoubles(0.1, 0.1, 9)) {
                for (double beta : Range.ofDoubles(5, 5, 5)) {

                    Bmf bnmf = new Bmf(numTopics, NUM_ITERS, alpha, beta);
                    bnmf.train();

                    double mae = QualityMeasures.MAE(bnmf);

                    System.out.println("numTopics = " + numTopics + "; alpha = " + alpha + "; beta = " + beta + "; mae = " + mae);

                    if (mae < minError) {
                        minError = mae;
                        best = "numTopics = " + numTopics + "; alpha = " + alpha + "; beta = " + beta + "; mae = " + mae;
                    }
                }
            }
        }

        System.out.println("\nBest result for BNMF => " + best);
    }
}