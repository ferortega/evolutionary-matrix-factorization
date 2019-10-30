package main.java.qualityMeasures;

import cf4j.Kernel;
import cf4j.TestUser;
import cf4j.model.matrixFactorization.FactorizationModel;

public class QualityMeasures {

    public static double MAE (FactorizationModel fm) {
        double sum = 0;
        int count = 0;

        for (TestUser user : Kernel.getInstance().getTestUsers()) {
            int userIndex = user.getUserIndex();

            for (int i = 0; i < user.getNumberOfTestRatings(); i++) {
                int itemCode = user.getTestItemAt(i);
                int itemIndex = Kernel.getInstance().getItemIndex(itemCode);

                double rating = user.getTestRatingAt(i);
                double prediction = fm.getPrediction(userIndex, itemIndex);

                sum += Math.abs(rating - prediction);
                count++;
            }
        }

        return  sum / count;
    }

    public static double MSE (FactorizationModel fm) {
        double sum = 0;
        int count = 0;

        for (TestUser user : Kernel.getInstance().getTestUsers()) {
            int userIndex = user.getUserIndex();

            for (int i = 0; i < user.getNumberOfTestRatings(); i++) {
                int itemCode = user.getTestItemAt(i);
                int itemIndex = Kernel.getInstance().getItemIndex(itemCode);

                double rating = user.getTestRatingAt(i);
                double prediction = fm.getPrediction(userIndex, itemIndex);

                sum += Math.pow(rating - prediction, 2);
                count++;
            }
        }

        return  sum / count;
    }

    public static double RMSE (FactorizationModel fm) {
        double sum = 0;
        int count = 0;

        for (TestUser user : Kernel.getInstance().getTestUsers()) {
            int userIndex = user.getUserIndex();

            for (int i = 0; i < user.getNumberOfTestRatings(); i++) {
                int itemCode = user.getTestItemAt(i);
                int itemIndex = Kernel.getInstance().getItemIndex(itemCode);

                double rating = user.getTestRatingAt(i);
                double prediction = fm.getPrediction(userIndex, itemIndex);

                sum += Math.pow(rating - prediction, 2);
                count++;
            }
        }

        return  Math.sqrt(sum / count);
    }
}
