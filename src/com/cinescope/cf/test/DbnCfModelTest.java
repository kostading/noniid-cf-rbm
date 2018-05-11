package com.cinescope.cf.test;

import com.cinescope.cf.EvalUtil;
import com.cinescope.cf.RbmBasedCF;
import com.cinescope.cf.TestResult;
import com.syvys.jaRBM.RBMNet;

public class DbnCfModelTest implements CfModelTest {
	@Override
	public TestResult execute(double[][] trainMatrix, double[][] testMatrix) {
		RBMNet rbmNet = RbmBasedCF.learnDBN(trainMatrix);
		
        double[][] hiddenActivities = rbmNet.getHiddenActivitiesFromVisibleData(trainMatrix);
        double[][] hiddenData = rbmNet.GenerateHiddenUnits(hiddenActivities);
        // generation phase
        double[][] visibleActivities = rbmNet.getVisibleActivitiesFromHiddenData(hiddenData);

        System.out.println("--- Evaluation ---");
        TestResult result = EvalUtil.evaluatePredictions(testMatrix, visibleActivities);
        EvalUtil.evaluatePerRatingPRAF(testMatrix, visibleActivities);
        
        return result;
	}
}
