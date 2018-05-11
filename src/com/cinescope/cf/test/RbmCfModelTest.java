package com.cinescope.cf.test;

import com.cinescope.cf.EvalUtil;
import com.cinescope.cf.MatrixUtil;
import com.cinescope.cf.RbmBasedCF;
import com.cinescope.cf.TestResult;
import com.syvys.jaRBM.RBM;

/**
 * Defines the test for the collaborative filtering (CF) model based on a modification
 * of Restricted Boltzmann Machines (RBM's) that that is able to model real-valued
 * user's ratings directly with its visible layer and also generate predictions for
 * the missing user's ratings as proposed in the master thesis.
 */
public class RbmCfModelTest implements CfModelTest {

	@Override
	public TestResult execute(double[][] trainMatrix, double[][] testMatrix) {
//		trainMatrix = MatrixUtil.toBinaryMatrix(trainMatrix);
		RBM userBasedRbm = RbmBasedCF.learnRBM(trainMatrix, null, 50, testMatrix);

        double[][] hiddenActivities = userBasedRbm.getHiddenActivitiesFromVisibleData(trainMatrix);
        double[][] hiddenData = userBasedRbm.GenerateHiddenUnits(hiddenActivities);
        // generation phase
        double[][] visibleActivities = userBasedRbm.getVisibleActivitiesFromHiddenData(hiddenData);
        double[][] visibleData1 = userBasedRbm.GenerateVisibleUnits(visibleActivities);
        
        System.out.println("--- Evaluation results ---");
//        visibleData1 = MatrixUtil.toRealMatrix(visibleData1);
        TestResult result = EvalUtil.evaluatePredictions(testMatrix, visibleData1);
        //EvalUtil.evaluatePerRatingPRAF(testMatrix, visibleData1);
        
        // item-based CF
        
//        trainMatrix = transposeMatrix(trainMatrix);
//        // trainMatrix = MatrixUtil.toBinaryMatrix(trainMatrix);
//        testMatrix = transposeMatrix(testMatrix);
//        
//        RBM itemBasedRbm = RbmBasedCF.learnRBM(trainMatrix, null, 150, testMatrix);
//
//        double[][] hiddenActivities = itemBasedRbm.getHiddenActivitiesFromVisibleData(trainMatrix);
//        double[][] hiddenData = itemBasedRbm.GenerateHiddenUnits(hiddenActivities);
//        // generation phase
//        double[][] visibleActivities = itemBasedRbm.getVisibleActivitiesFromHiddenData(hiddenData);
//        double[][] visibleData2 = itemBasedRbm.GenerateVisibleUnits(visibleActivities);
        // visibleData2 = MatrixUtil.toRealMatrix(visibleData2);
        
//        System.out.println("--- Evaluation results ---");
//        TestResult result = EvalUtil.evaluatePredictions(testMatrix, visibleData2);
//        EvalUtil.evaluatePerRatingPRAF(transposedTestMatrix, visibleData2);
//        
//        // combined CF
//        
//        double[][] transposedVisibleData2 = transposeMatrix(visibleData2);
//        double[][] combined = average(visibleData1, transposedVisibleData2);
//        
//        System.out.println("--- Evaluation results ---");
//        result = EvalUtil.evaluatePredictions(testMatrix, combined);
//        EvalUtil.evaluatePerRatingPRAF(testMatrix, combined);
        
		return result;
	}

	@Override
	public String toString() {
		return "Real-valued RBM based CF";
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
