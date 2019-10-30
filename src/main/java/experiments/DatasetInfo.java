package main.java.experiments;

import cf4j.Kernel;

public class DatasetInfo {

    private static final String BINARY_FILE = "datasets/ml100k.cf4j";
//    private static final String BINARY_FILE = "datasets/filmtrust.cf4j";

    public static void main(String[] args) {
        Kernel.getInstance().readKernel(BINARY_FILE);
        System.out.println(Kernel.getInstance().getKernelInfo());
    }
}
