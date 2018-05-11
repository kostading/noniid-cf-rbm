package com.cinescope.cf.test;

import java.text.DecimalFormat;

import com.cinescope.cf.EvalUtil;
import com.cinescope.cf.HybridRbmLearner2;
import com.cinescope.cf.TestResult;
import com.syvys.jaRBM.RBM;
import com.syvys.jaRBM.RBMImpl;
import com.syvys.jaRBM.Layers.Layer;
import com.syvys.jaRBM.Layers.LinearLayer;
import com.syvys.jaRBM.Layers.LogisticLayer;
import com.syvys.jaRBM.Math.Matrix;

public class HybridRbmCfModelTest2 implements CfModelTest {
	@Override
	public TestResult execute(double[][] trainMatrix, double[][] testMatrix) {
    	double[][] transposedData = Matrix.getTranspose(trainMatrix);
        
        Layer userVisibleLayer = new LinearLayer(trainMatrix[0].length);
        Layer userHiddenLayer = new LogisticLayer(50);
        RBM userBasedRbm = new RBMImpl(userVisibleLayer, userHiddenLayer);
        
        Layer userVisibleLayer2 = new LinearLayer(trainMatrix[0].length);
        Layer userHiddenLayer2 = new LogisticLayer(10);
        RBM userBasedRbm2 = new RBMImpl(userVisibleLayer2, userHiddenLayer2);
        
        Layer itemVisibleLayer = new LinearLayer(trainMatrix.length);
        Layer itemHiddenLayer = new LogisticLayer(130);
        RBM itemBasedRbm = new RBMImpl(itemVisibleLayer, itemHiddenLayer);
     
        for (int epochs = 1; epochs <= 300; epochs++) {
            double error = HybridRbmLearner2.Learn(userBasedRbm, userBasedRbm2, itemBasedRbm, trainMatrix, transposedData);
            if (epochs % 50 == 0) {
            	double[][] visibleData = generateVisibleData(userBasedRbm, userBasedRbm2, itemBasedRbm, trainMatrix);
            	System.out.println("\nEpochs: " + epochs + ", results:");
            	EvalUtil.evaluatePredictions(testMatrix, visibleData);
            	EvalUtil.evaluatePerRatingPRAF(testMatrix, visibleData);
            }
        	DecimalFormat df = new DecimalFormat("#.#####");
        	System.out.println("Epoch: " + epochs + "; Reconstruction error: " + df.format(error));
        }
        
        System.out.println("--- Evaluation results ---");
    	double[][] visibleData = generateVisibleData(userBasedRbm, userBasedRbm2, itemBasedRbm, trainMatrix);
    	
    	TestResult result = EvalUtil.evaluatePredictions(testMatrix, visibleData);
        EvalUtil.evaluatePerRatingPRAF(testMatrix, visibleData);
        
        System.out.println();
        
        return result;
	}

    private static int[] getRatingsPerUser(double[][] data) {
    	int[] ratingsPerUser = new int[data.length];
    	
        for (int i=0; i<data.length; i++) {
        	int ratingsCount = 0;
        	
        	for (int j=0; j<data[i].length; j++) {
        		if (data[i][j] > 0) {
        			ratingsCount++;
        		}
        	}
        	
        	ratingsPerUser[i] = ratingsCount;
        }
        
        return ratingsPerUser;
    }
    
    private static double getAverageRatingsPerUser(int[] ratingsPerUser) {
    	int totalRatings = 0;
        for (int i=0; i<ratingsPerUser.length; i++) {
        	totalRatings += ratingsPerUser[i];
        }
        
        return (double) totalRatings / ratingsPerUser.length;
    }
	
	private static double[][] generateVisibleData(RBM userBasedRbm, RBM userBasedRbm2, RBM itemBasedRbm, double[][] trainMatrix) {
        // get data from the user-based RBM
        double[][] userHiddenActivities = userBasedRbm.getHiddenActivitiesFromVisibleData(trainMatrix);
        double[][] userHiddenData = userBasedRbm.GenerateHiddenUnits(userHiddenActivities);
        // generation phase
        double[][] userVisibleActivities = userBasedRbm.getVisibleActivitiesFromHiddenData(userHiddenData);
        double[][] userVisibleData = userBasedRbm.GenerateVisibleUnits(userVisibleActivities);
        
        // get data from the user-based RBM 2
        double[][] userHiddenActivities2 = userBasedRbm2.getHiddenActivitiesFromVisibleData(trainMatrix);
        double[][] userHiddenData2 = userBasedRbm2.GenerateHiddenUnits(userHiddenActivities2);
        // generation phase
        double[][] userVisibleActivities2 = userBasedRbm2.getVisibleActivitiesFromHiddenData(userHiddenData2);
        double[][] userVisibleData2 = userBasedRbm2.GenerateVisibleUnits(userVisibleActivities2);
        
        // get data from the item-based RBM
        double[][] itemHiddenActivities = itemBasedRbm.getHiddenActivitiesFromVisibleData(Matrix.getTranspose(trainMatrix));
        double[][] itemHiddenData = itemBasedRbm.GenerateHiddenUnits(itemHiddenActivities);
        // generation phase
        double[][] itemVisibleActivities = itemBasedRbm.getVisibleActivitiesFromHiddenData(itemHiddenData);
        double[][] itemVisibleData = itemBasedRbm.GenerateVisibleUnits(itemVisibleActivities);
        
        // combine the data from the user-based and item-based RBM's
        double[][] visibleData = new double[userVisibleData.length][userVisibleData[0].length];
        for (int i=0; i<visibleData.length; i++) {
        	for (int j=0; j<visibleData[i].length; j++) {
        		visibleData[i][j] = (userVisibleData[i][j] + itemVisibleData[j][i] + userVisibleData2[i][j]) / 3.0;
        	}
        }
        
        return visibleData;
	}
	
	@Override
	public String toString() {
		return "Hybrid RBM based CF 2";
	}

}
