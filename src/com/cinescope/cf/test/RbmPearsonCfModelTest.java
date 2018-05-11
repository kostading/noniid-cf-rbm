package com.cinescope.cf.test;

import com.cinescope.cf.CorrelationUtil;
import com.cinescope.cf.EvalUtil;
import com.cinescope.cf.MemoryBasedCF;
import com.cinescope.cf.RbmBasedCF;
import com.cinescope.cf.TestResult;
import com.syvys.jaRBM.RBM;

/**
 * Defines the test for the collaborative filtering (CF) model based on our original
 * modification to the classical Pearson-based algorithm proposed in this master
 * thesis that works with both original user-item matrix data and the imputed
 * user-item matrix produced by the RBM CF model.
 */
public class RbmPearsonCfModelTest implements CfModelTest {
	@Override
	public TestResult execute(double[][] trainMatrix, double[][] testMatrix) {
//		RBM userBasedRbm = RbmBasedCF.learnRBM(trainMatrix, null, 50);
//
//        double[][] hiddenActivities = userBasedRbm.getHiddenActivitiesFromVisibleData(trainMatrix);
//        double[][] hiddenData = userBasedRbm.GenerateHiddenUnits(hiddenActivities);
//        // generation phase
//        double[][] visibleActivities = userBasedRbm.getVisibleActivitiesFromHiddenData(hiddenData);
//        double[][] visibleData1 = userBasedRbm.GenerateVisibleUnits(visibleActivities);
        
        // item-based CF
        
        double[][] transposedTrainMatrix = transposeMatrix(trainMatrix);
        
        RBM itemBasedRbm = RbmBasedCF.learnRBM(transposedTrainMatrix, null, 150, transposeMatrix(testMatrix));

        double[][] hiddenActivities = itemBasedRbm.getHiddenActivitiesFromVisibleData(transposedTrainMatrix);
        double[][] hiddenData = itemBasedRbm.GenerateHiddenUnits(hiddenActivities);
        // generation phase
        double[][] visibleActivities = itemBasedRbm.getVisibleActivitiesFromHiddenData(hiddenData);
        double[][] visibleData2 = itemBasedRbm.GenerateVisibleUnits(visibleActivities);
        
        // combined CF
        
        double[][] transposedVisibleData2 = transposeMatrix(visibleData2);
//        double[][] combined = average(visibleData1, transposedVisibleData2);
//        
//        System.out.println("--- Evaluation results ---");
//        TestResult result = EvalUtil.evaluatePredictions(testMatrix, combined);
//        EvalUtil.evaluatePerRatingPRAF(testMatrix, combined);
        
        // Correlation
        fixRatings(trainMatrix, transposedVisibleData2);
        
        System.out.println("--- Calculating Item-Based Pearson CF ---");
        double[][] visibleData = CorrelationUtil.applyPearsonFiltering(trainMatrix, transposedVisibleData2);
        
//        System.out.println("--- Calculating Item-Based Cosine CF ---");
//        double[][] visibleData = MemoryBasedCF.applyItemCosineFiltering(trainMatrix, transposedVisibleData2);
        
        System.out.println("--- Evaluation results ---");
        TestResult result = EvalUtil.evaluatePredictions(testMatrix, visibleData);
        //EvalUtil.evaluatePerRatingPRAF(testMatrix, visibleData);
        
        return result;
	}
	
	@Override
	public String toString() {
		return "Hybrid Pearson CF + real-valued RBM based CF";
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
