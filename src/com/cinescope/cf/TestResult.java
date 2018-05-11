package com.cinescope.cf;

import java.text.DecimalFormat;

/**
 * Contains the values of various evaluation metrics that resulted from an
 * experiment with a collaborative filtering (CF) model.
 */
public class TestResult {
	public int numResults = 0;
	
	public int numRatings = 0;
	
	public double MAE = 0;
	
	public double RMSE = 0;
	
	public double sdMAE = 0;
	
	public double precision = 0;
	
	public double recall = 0;
	
	public double accuracy = 0;
	
	/**
	 * Accumulates the provided test results to the current test result's values.
	 */
	public void accumulateResult(TestResult testResult) {
		numResults++;
		
		numRatings = (numRatings * (numResults-1) + testResult.numRatings) / numResults;
		MAE = (MAE * (numResults-1) + testResult.MAE) / numResults;
		RMSE = (RMSE * (numResults-1) + testResult.RMSE) / numResults;
		sdMAE = (sdMAE * (numResults-1) + testResult.sdMAE) / numResults;
		precision = (precision * (numResults-1) + testResult.precision) / numResults;
		recall = (recall * (numResults-1) + testResult.recall) / numResults;
		accuracy = (accuracy * (numResults-1) + testResult.accuracy) / numResults;
	}
	
	public void printTotals() {
		DecimalFormat df = new DecimalFormat("#.#####");
		System.out.println("\nTotal MAE: " + df.format(MAE));
		System.out.println("\nTotal NMAE: " + df.format((double) MAE / 4));
		System.out.println("\nTotal RMSE: " + df.format(RMSE));
		double hw = 1.96 * sdMAE / Math.sqrt(numResults * numRatings);
		System.out.println("Total MAE Confidence Interval: [" + df.format(MAE - hw) + "; " + df.format(MAE + hw) + "]");
		System.out.println("\nTotal precision: " + df.format(precision));
		System.out.println("Total recall: " + df.format(recall));
		System.out.println("Total accuracy: " + df.format(accuracy));
		System.out.println("Total F-measure: " + df.format(2 * precision * recall / (precision + recall)));	
	}

}
