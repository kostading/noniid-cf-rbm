package com.cinescope.cf.test;

import com.cinescope.cf.CorrelationUtil;
import com.cinescope.cf.EvalUtil;
import com.cinescope.cf.HybridRbmLearner;
import com.cinescope.cf.RbmBasedCF;
import com.cinescope.cf.TestResult;
import com.syvys.jaRBM.RBM;
import com.syvys.jaRBM.RBMImpl;
import com.syvys.jaRBM.Layers.Layer;
import com.syvys.jaRBM.Layers.LinearLayer;
import com.syvys.jaRBM.Layers.LogisticLayer;
import com.syvys.jaRBM.Math.Matrix;

public class CombinedCfCorrInModelTest implements CfModelTest {
	@Override
	public TestResult execute(double[][] trainMatrix, double[][] testMatrix) {
    	double[][] transposedData = Matrix.getTranspose(trainMatrix);
        Layer userVisibleLayer = new LinearLayer(trainMatrix[0].length);
        Layer userHiddenLayer = new LogisticLayer(150);
        RBM userBasedRbm = new RBMImpl(userVisibleLayer, userHiddenLayer);
        
        Layer itemVisibleLayer = new LinearLayer(trainMatrix.length);
        Layer itemHiddenLayer = new LogisticLayer(130);
        RBM itemBasedRbm = new RBMImpl(itemVisibleLayer, itemHiddenLayer);
     
        for (int epochs = 1; epochs <= 300; epochs++) {
            double error = HybridRbmLearner.Learn(userBasedRbm, itemBasedRbm, trainMatrix, transposedData);
        }
        
        System.out.println("--- Evaluation results ---");
    	double[][] visibleData = generateVisibleData(userBasedRbm, itemBasedRbm, trainMatrix);
    	
    	TestResult result = EvalUtil.evaluatePredictions(testMatrix, visibleData);
        
        return result;
	}
	
	private static double[][] generateVisibleData(RBM userBasedRbm, RBM itemBasedRbm, double[][] trainMatrix) {
        // get data from the user-based RBM
        double[][] userHiddenActivities = userBasedRbm.getHiddenActivitiesFromVisibleData(trainMatrix);
        double[][] userHiddenData = userBasedRbm.GenerateHiddenUnits(userHiddenActivities);
        // generation phase
        double[][] userVisibleActivities = userBasedRbm.getVisibleActivitiesFromHiddenData(userHiddenData);
        double[][] userVisibleData = userBasedRbm.GenerateVisibleUnits(userVisibleActivities);
        
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
        		visibleData[i][j] = (userVisibleData[i][j] + itemVisibleData[j][i]) / 2.0;
        	}
        }
        
        fixRatings(trainMatrix, visibleData);
        CorrelationUtil.applyPearsonFiltering(trainMatrix, visibleData);
        
        return visibleData;
	}
	
	@Override
	public String toString() {
		return "CombinedCfCorrInModelTest";
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
	
	private static double[][] transposeMatrix(double[][] original) {
		double[][] transposed = new double[original[0].length][original.length];
		
		for (int i=0; i<original.length; i++) {
			for (int j=0; j<original[i].length; j++) {
				transposed[j][i] = original[i][j];
			}
		}
		
		return transposed;
	}
	
	private static double[][] average(double[][] a, double[][] b) {
		double[][] result = new double[a.length][a[0].length];
		
		for (int i=0; i<a.length; i++) {
			for (int j=0; j<a[i].length; j++) {
				result[i][j] = (a[i][j] + b[i][j]) / 2.0;
			}
		}
		
		return result;
	}

}
