package main.java.experiments;

import cf4j.Kernel;

public class GenerateBinaryFiles {

    private static final String DATASET = "datasets/ml100k.dat";
    private static final String DATASET_SEPARATOR = "::";

    private static final double TEST_USERS = 0.2;
    private static final double TEST_ITEMS = 0.2;

    private static final String BINARY_FILE = "datasets/ml100k.cf4j";

//    private static final String DATASET = "datasets/filmtrust.txt";
//    private static final String DATASET_SEPARATOR = " ";
//
//    private static final double TEST_USERS = 0.2;
//    private static final double TEST_ITEMS = 0.2;
//
//    private static final String BINARY_FILE = "datasets/filmtrust.cf4j";

    public static void main (String [] args) {
        Kernel.getInstance().open(DATASET, TEST_USERS, TEST_ITEMS, DATASET_SEPARATOR);
        Kernel.getInstance().writeKernel(BINARY_FILE);
    }
}
