package com.cinescope.cf.test;

import java.text.DecimalFormat;

import com.cinescope.cf.CombinedHybridRbmLearner;
import com.cinescope.cf.EvalUtil;
import com.cinescope.cf.HybridRbmLearner;
import com.cinescope.cf.MemoryBasedCF;
import com.cinescope.cf.RbmBasedCF;
import com.cinescope.cf.TestResult;
import com.syvys.jaRBM.RBM;
import com.syvys.jaRBM.RBMImpl;
import com.syvys.jaRBM.Layers.Layer;
import com.syvys.jaRBM.Layers.LinearLayer;
import com.syvys.jaRBM.Layers.LogisticLayer;
import com.syvys.jaRBM.Math.Matrix;

public class CombinedHybridRbmCfModelTest implements CfModelTest {
	public static double[][] nbPredictions;
	public static int[] ratingsPerItem;
	
	@Override
	public TestResult execute(double[][] trainMatrix, double[][] testMatrix) {
		ratingsPerItem = getRatingsPerItem(trainMatrix);
		nbPredictions = MemoryBasedCF.applyItemCosineFiltering(trainMatrix, trainMatrix);
		//fixRatings(trainMatrix, nbPredictions);
		EvalUtil.evaluatePredictions(testMatrix, nbPredictions);
        // null values missing in original data
        // RbmBasedCF.nullMissingValues(trainMatrix, nbPredictions);
        
    	double[][] transposedData = Matrix.getTranspose(trainMatrix);
    	
        Layer userVisibleLayer = new LinearLayer(trainMatrix[0].length);
        Layer userHiddenLayer = new LogisticLayer(150);
        RBM userBasedRbm = new RBMImpl(userVisibleLayer, userHiddenLayer);
        
        Layer itemVisibleLayer = new LinearLayer(trainMatrix.length);
        Layer itemHiddenLayer = new LogisticLayer(130);
        RBM itemBasedRbm = new RBMImpl(itemVisibleLayer, itemHiddenLayer);
     
        for (int epochs = 1; epochs <= 700; epochs++) {
            double error = HybridRbmLearner.Learn(userBasedRbm, itemBasedRbm, trainMatrix, transposedData);
            if (epochs % 50 == 0) {
            	double[][] visibleData = generateVisibleData(userBasedRbm, itemBasedRbm, trainMatrix, transposedData);
            	System.out.println("\nEpochs: " + epochs + ", results:");
            	EvalUtil.evaluatePredictions(testMatrix, visibleData);
            }
        	DecimalFormat df = new DecimalFormat("#.#####");
        	System.out.println("Epoch: " + epochs + "; Reconstruction error: " + df.format(error));
        }
        
        System.out.println("--- Evaluation results ---");
    	double[][] visibleData = generateVisibleData(userBasedRbm, itemBasedRbm, trainMatrix, transposedData);
    	
    	TestResult result = EvalUtil.evaluatePredictions(testMatrix, visibleData);
       // EvalUtil.evaluatePerRatingPRAF(testMatrix, visibleData);
        
        System.out.println();
        
        return result;
	}
	
	private static double[][] generateVisibleData(RBM userBasedRbm, RBM itemBasedRbm, double[][] trainMatrix, double[][] transposedData) {
        // get data from the user-based RBM
        double[][] userHiddenActivities = userBasedRbm.getHiddenActivitiesFromVisibleData(trainMatrix);
        double[][] userHiddenData = userBasedRbm.GenerateHiddenUnits(userHiddenActivities);
        // generation phase
        double[][] userVisibleActivities = userBasedRbm.getVisibleActivitiesFromHiddenData(userHiddenData);
        double[][] userVisibleData = userBasedRbm.GenerateVisibleUnits(userVisibleActivities);
        
        // get data from the item-based RBM
        double[][] itemHiddenActivities = itemBasedRbm.getHiddenActivitiesFromVisibleData(transposedData);
        double[][] itemHiddenData = itemBasedRbm.GenerateHiddenUnits(itemHiddenActivities);
        // generation phase
        double[][] itemVisibleActivities = itemBasedRbm.getVisibleActivitiesFromHiddenData(itemHiddenData);
        double[][] itemVisibleData = itemBasedRbm.GenerateVisibleUnits(itemVisibleActivities);
        
        // combine the data from the user-based and item-based RBM's
        double[][] visibleData = new double[userVisibleData.length][userVisibleData[0].length];
        for (int i=0; i<visibleData.length; i++) {
        	for (int j=0; j<visibleData[i].length; j++) {
        		if (ratingsPerItem[j] <= 50) {
        			visibleData[i][j] = nbPredictions[i][j];
        		} else {
        			visibleData[i][j] = (userVisibleData[i][j] + itemVisibleData[j][i]) / 2.0;
        		}
        	}
        }
        
        return visibleData;
	}
	
	private static void fixRatings(double[][] originalMatrix, double[][] imputedMatrix) {
		for (int user = 0; user < originalMatrix.length; user++) {
			for (int item = 0; item < originalMatrix[user].length; item++) {
				if (originalMatrix[user][item] != 0) {
					imputedMatrix[user][item] = originalMatrix[user][item];
				}
			}
		}
	}
	
    private static int[] getRatingsPerItem(double[][] data) {
    	int[] ratingsPerItem = new int[data[0].length];
    	
        for (int item=0; item<data[0].length; item++) {
        	int ratingsCount = 0;
        	
        	for (int user=0; user<data.length; user++) {
        		if (data[user][item] > 0) {
        			ratingsCount++;
        		}
        	}
        	
        	ratingsPerItem[item] = ratingsCount;
        }
        
        return ratingsPerItem;
    }
	
	@Override
	public String toString() {
		return "Hybrid RBM based CF";
	}

}
