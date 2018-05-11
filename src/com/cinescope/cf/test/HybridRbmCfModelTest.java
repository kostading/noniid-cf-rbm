package com.cinescope.cf.test;

import java.text.DecimalFormat;

import com.cinescope.cf.EvalUtil;
import com.cinescope.cf.HybridRbmLearner;
import com.cinescope.cf.MemoryBasedCF;
import com.cinescope.cf.TestResult;
import com.syvys.jaRBM.RBM;
import com.syvys.jaRBM.RBMImpl;
import com.syvys.jaRBM.Layers.Layer;
import com.syvys.jaRBM.Layers.LinearLayer;
import com.syvys.jaRBM.Layers.LogisticLayer;
import com.syvys.jaRBM.Layers.StochasticBinaryLayer;
import com.syvys.jaRBM.Layers.StochasticLinearLayer;
import com.syvys.jaRBM.Math.Matrix;

public class HybridRbmCfModelTest implements CfModelTest {
	public static int[] ratingsPerItem;
	
	public static double[][] userSimilarity;
	public static double[][] itemSimilarity;
	
	@Override
	public TestResult execute(double[][] trainMatrix, double[][] testMatrix) {
//		userSimilarity = MemoryBasedCF.calcualteUserSimilarity(trainMatrix);
//		itemSimilarity =  MemoryBasedCF.calcualteItemSimilarity(trainMatrix, trainMatrix);
		
    	double[][] transposedData = Matrix.getTranspose(trainMatrix);
		//ratingsPerItem = getRatingsPerItem(trainMatrix);
    	//System.out.println("User filter");
        //filterUserItemMatrix(trainMatrix);
        //fitlerUsers = true;
        //System.out.println("Item filter");
        //filterUserItemMatrix(transposedData);
    	
        Layer userVisibleLayer = new LinearLayer(trainMatrix[0].length);
        Layer userHiddenLayer = new StochasticBinaryLayer(CfModelTestExecutor.USER_HIDDEN_SIZE);
        RBM userBasedRbm = new RBMImpl(userVisibleLayer, userHiddenLayer);
        
        Layer itemVisibleLayer = new LinearLayer(trainMatrix.length);
        Layer itemHiddenLayer = new StochasticBinaryLayer(CfModelTestExecutor.ITEM_HIDDEN_SIZE);
        RBM itemBasedRbm = new RBMImpl(itemVisibleLayer, itemHiddenLayer);
     
        for (int epochs = 1; epochs <= 1000; epochs++) {
            double error = HybridRbmLearner.Learn(userBasedRbm, itemBasedRbm, trainMatrix, transposedData);
            if (epochs % 50 == 0) {
            	double[][] visibleData = generateVisibleData(userBasedRbm, itemBasedRbm, trainMatrix, transposedData);
            	System.out.print("\nEpochs: " + epochs + ", results: ");
            	EvalUtil.evaluatePredictions(testMatrix, visibleData);
            }
//        	DecimalFormat df = new DecimalFormat("#.#####");
//        	System.out.println("Epoch: " + epochs + "; Reconstruction error: " + df.format(error));
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
        //double[][] userVisibleData = userBasedRbm.GenerateVisibleUnits(userVisibleActivities);
        
        // get data from the item-based RBM
        double[][] itemHiddenActivities = itemBasedRbm.getHiddenActivitiesFromVisibleData(transposedData);
        double[][] itemHiddenData = itemBasedRbm.GenerateHiddenUnits(itemHiddenActivities);
        // generation phase
        double[][] itemVisibleActivities = itemBasedRbm.getVisibleActivitiesFromHiddenData(itemHiddenData);
        //double[][] itemVisibleData = itemBasedRbm.GenerateVisibleUnits(itemVisibleActivities);
        
        // combine the data from the user-based and item-based RBM's
        double[][] visibleData = new double[userVisibleActivities.length][userVisibleActivities[0].length];
        for (int i=0; i<visibleData.length; i++) {
        	for (int j=0; j<visibleData[i].length; j++) {
        		visibleData[i][j] = (userVisibleActivities[i][j] + itemVisibleActivities[j][i]) / 2.0;
        	}
        }
        visibleData = userBasedRbm.GenerateVisibleUnits(visibleData);
        
        return visibleData;
	}
	
	private static void filterUserItemMatrix(double[][] userItemMatrix) {
		ratingsPerItem = getRatingsPerItem(userItemMatrix);
		int numFiltered = 0;
		
        for (int i=0; i<userItemMatrix.length; i++) {
        	for (int j=0; j<userItemMatrix[i].length; j++) {
        		if (ratingsPerItem[j] < 20 && userItemMatrix[i][j] != 0) {
        			userItemMatrix[i][j] = 0;
        			numFiltered++;
        		}
        	}	
        }
        
		System.out.println("Num filtered: " + numFiltered);
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
		return "Real UI-RBM";
	}

}
