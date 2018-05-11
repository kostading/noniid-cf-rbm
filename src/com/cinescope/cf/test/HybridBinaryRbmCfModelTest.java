package com.cinescope.cf.test;

import java.text.DecimalFormat;

import com.cinescope.cf.HybridBinaryRbmLearner2;
import com.cinescope.cf.HybridBinaryRbmLearner;
import com.cinescope.cf.CfSoftmaxLayer;
import com.cinescope.cf.EvalUtil;
import com.cinescope.cf.HybridRbmLearner;
import com.cinescope.cf.MatrixUtil;
import com.cinescope.cf.RbmBasedCF;
import com.cinescope.cf.TestResult;
import com.syvys.jaRBM.RBM;
import com.syvys.jaRBM.RBMImpl;
import com.syvys.jaRBM.Layers.Layer;
import com.syvys.jaRBM.Layers.LinearLayer;
import com.syvys.jaRBM.Layers.LogisticLayer;
import com.syvys.jaRBM.Math.Matrix;

public class HybridBinaryRbmCfModelTest implements CfModelTest {
	
	@Override
	public TestResult execute(double[][] trainMatrix, double[][] testMatrix) {
    	double[][] transposedData = Matrix.getTranspose(trainMatrix);
    	trainMatrix = MatrixUtil.toBinaryMatrix(trainMatrix);
    	transposedData = MatrixUtil.toBinaryMatrix(transposedData);
    	
        Layer userVisibleLayer = new CfSoftmaxLayer(trainMatrix[0].length);
        Layer userHiddenLayer = new LogisticLayer(CfModelTestExecutor.USER_HIDDEN_SIZE);
        RBM userBasedRbm = new RBMImpl(userVisibleLayer, userHiddenLayer);
        
        Layer itemVisibleLayer = new CfSoftmaxLayer(transposedData[0].length);
        Layer itemHiddenLayer = new LogisticLayer(CfModelTestExecutor.ITEM_HIDDEN_SIZE);
        RBM itemBasedRbm = new RBMImpl(itemVisibleLayer, itemHiddenLayer);
     
        for (int epochs = 1; epochs <= 1000; epochs++) {
            double error = HybridBinaryRbmLearner.Learn(userBasedRbm, itemBasedRbm, trainMatrix, transposedData);
            if (epochs % 50 == 0) {
            	double[][] visibleData = generateVisibleData(userBasedRbm, itemBasedRbm, trainMatrix, transposedData);
//            	System.out.println("\nEpochs: " + epochs + ", results:");
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
		
        double[][] userSum = userBasedRbm.getHiddenActivitiesFromVisibleData(trainMatrix);
        userSum = userBasedRbm.GenerateHiddenUnits(userSum);
        userSum  = userBasedRbm.getDownwardSWSum(userSum);
        // item-based
        double[][] itemSum = itemBasedRbm.getHiddenActivitiesFromVisibleData(transposedData);
        itemSum = itemBasedRbm.GenerateHiddenUnits(itemSum);
        itemSum  = itemBasedRbm.getDownwardSWSum(itemSum);
        
        for (int i=0; i<userSum.length; i++) {
        	for (int j=0; j<userSum[i].length/5; j++) {
        		for (int k=0; k<5; k++) {
        			double newVal = (userSum[i][j*5+k] + itemSum[j][i*5+k]) / 2.0;
        			userSum[i][j*5+k] = newVal;
        			itemSum[j][i*5+k] = newVal;
        		}
        	}
        }
        double[][] visibleData = userBasedRbm.getVisibleLayer().getActivationProbabilities(userSum);
        visibleData = userBasedRbm.GenerateVisibleUnits(visibleData);
        visibleData = MatrixUtil.toRealMatrix(visibleData);
        
        return visibleData;
	}
	
	@Override
	public String toString() {
		return "Hybrid Binary RBM based CF";
	}

}
