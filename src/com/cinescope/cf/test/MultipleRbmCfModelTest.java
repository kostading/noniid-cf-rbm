package com.cinescope.cf.test;

import java.text.DateFormat;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import com.cinescope.cf.CorrelationUtil;
import com.cinescope.cf.EvalUtil;
import com.cinescope.cf.HybridRbmLearner;
import com.cinescope.cf.TestResult;
import com.syvys.jaRBM.RBM;
import com.syvys.jaRBM.RBMImpl;
import com.syvys.jaRBM.Layers.Layer;
import com.syvys.jaRBM.Layers.LinearLayer;
import com.syvys.jaRBM.Layers.LogisticLayer;
import com.syvys.jaRBM.Math.Matrix;

public class MultipleRbmCfModelTest implements CfModelTest {
	@Override
	public TestResult execute(double[][] trainMatrix, double[][] testMatrix) {
    	double[][] transposedData = Matrix.getTranspose(trainMatrix);
		double[][] predictions = new double[trainMatrix.length][trainMatrix[0].length];
		double[][] userCorrelations = CorrelationUtil.calculateUserCorrelations(trainMatrix);
		
		int user;
		for (user=0; user<=10; user++) {
			List<double[]> correlatedTrainMatrix = new ArrayList<double[]>();
			int userIndex = 0;
			
			for (int otherUser=0; otherUser<trainMatrix.length; otherUser++) {
				if (otherUser == user) {
					correlatedTrainMatrix.add(trainMatrix[otherUser]);
					userIndex = correlatedTrainMatrix.size()-1;
				} else if (userCorrelations[user][otherUser] >= 0.4) {
					correlatedTrainMatrix.add(trainMatrix[otherUser]);
				}
			}
			
			double[][] userTrainMatrix =
					correlatedTrainMatrix.toArray(new double[correlatedTrainMatrix.size()][trainMatrix[0].length]);
			
			DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
			Date date = new Date();
			System.out.println(dateFormat.format(date));
			System.out.println("\nUser: " + user + " | Correlated users: " + userTrainMatrix.length);
			
	        Layer userVisibleLayer = new LinearLayer(userTrainMatrix[0].length);
	        Layer userHiddenLayer = new LogisticLayer(150);
	        RBM userBasedRbm = new RBMImpl(userVisibleLayer, userHiddenLayer);
	        
	        Layer itemVisibleLayer = new LinearLayer(userTrainMatrix.length);
	        Layer itemHiddenLayer = new LogisticLayer(20);
	        RBM itemBasedRbm = new RBMImpl(itemVisibleLayer, itemHiddenLayer);
	     
	        for (int epochs = 1; epochs <= 200; epochs++) {
	            double error = HybridRbmLearner.Learn(userBasedRbm, itemBasedRbm, userTrainMatrix, transposedData);
//	            System.out.print(" " + epochs);
	        }
	      
	    	double[][] visibleData = generateVisibleData(userBasedRbm, itemBasedRbm, userTrainMatrix);
	    	predictions[user] = visibleData[userIndex];
		}
		
    	EvalUtil.evaluateMAE(testMatrix, predictions, user);

    	//TestResult result = EvalUtil.evaluatePredictions(testMatrix, predictions);
        System.out.println();
        
        return new TestResult();
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
        
        return visibleData;
	}
	
	@Override
	public String toString() {
		return "Multiple RBM based CF";
	}

}

