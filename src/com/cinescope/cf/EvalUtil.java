package com.cinescope.cf;

import java.text.DecimalFormat;

import com.cinescope.cf.test.HybridRbmCfModelTest;

/**
 * Contains various utilities for evaluation of collaborative filtering models.
 */
public final class EvalUtil {

	/**
	 * Evaluates the provided users' ratings predictions against the metrics precision,
	 * recall, accuracy and F1.
	 */
	public static TestResult evaluatePRAF(double[][] expectations, double[][] predictions) {
		int numRatings = 0;
		int truePos = 0; int trueNeg = 0;
		int falsePos = 0; int falseNeg = 0;
		
		int error5 = 0; int count5=0;
		int error4 = 0; int count4=0;
		
		for (int i=0; i<expectations.length; i++) {
			for (int j=0; j<expectations[i].length; j++) {
				if (expectations[i][j] == 0.0) {
					continue;
				}
				numRatings++;
				long expected = Math.round(expectations[i][j]);
				long predicted = Math.round(predictions[i][j]);
				
				// Count the expected ratings 4 and 5 ratings that were
				// incorrectly assigned negative class.
				if (expected == 5) {
					if (predicted <=3) {
						error5++;
					}
					count5++;
				} else if (expected == 4) {
					if (predicted <=3) {
						error4++;
					}
					count4++;
				}
				
				if (expected > 3 && predicted > 3) {
					truePos++;
				} else if (expected <= 3 && predicted <= 3) {
					trueNeg++;
				} else if (expected > 3 && predicted <= 3) {
					falseNeg++;
				} else if (expected <= 3 && predicted > 3) {
					falsePos++;
				}
			}
		}
		
		TestResult result = evaluatePRAF(truePos, falsePos, trueNeg, falseNeg);
		result.numRatings = numRatings;
		
		DecimalFormat df = new DecimalFormat("#.#####");
//		System.out.println("\nTrue ratings 5 incorrectly assigned negative: " +
//				df.format((double)error5/count5));
//		System.out.println("True ratings 4 incorrectly assigned negative: " +
//				df.format((double)error4/count4));
		
		return result;
	}
	
	/**
	 * Evaluates the provided users' ratings predictions against the metrics precision,
	 * recall, accuracy, F1 and MAE.
	 */
	public static TestResult evaluatePredictions(double[][] expectations, double[][] predictions) {
		TestResult result = evaluateMAE(expectations, predictions);
		result.sdMAE = calculateSdMAE(result.MAE, expectations, predictions);
		//printConfidenceInterval(result);
		
//		TestResult finalResult = evaluatePRAF(expectations, predictions);
//		finalResult.MAE = result.MAE;
//		finalResult.RMSE = result.RMSE;
//		finalResult.sdMAE = result.sdMAE;

		return result;
	}
	
	/**
	 * Evaluates the provided users' ratings predictions for each individual rating
	 * from 1 to 5 against the metrics precision, recall, accuracy and F1.
	 */
	public static void evaluatePerRatingPRAF(double[][] expectations, double[][] predictions) {		
		int[] truePos = new int[6];
		int[] falsePos = new int[6];
		int[] trueNeg = new int[6];
		int[] falseNeg = new int[6];
		
		for (int i=0; i<expectations.length; i++) {
			for (int j=0; j<expectations[i].length; j++) {
				if (expectations[i][j] == 0.0) {
					continue;
				}
				long expected = Math.round(expectations[i][j]);
				long predicted = Math.round(predictions[i][j]);
				
				for (int k=1; k<=5; k++) {
					if (expected == k) {
						if (predicted == k) {
							truePos[k]++;
						} else {
							falseNeg[k]++;
						}
					} else if (predicted == k) {
						falsePos[k]++;
					} else {
						trueNeg[k]++;
					}
				}
			}
		}
		
		for (int k=1; k<=5; k++) {
			System.out.println("\nEvaluation on individual rating: " + k);
			evaluatePRAF(truePos[k], falsePos[k], trueNeg[k], falseNeg[k]);
		}
	}
	
	/**
	 * Evaluates the provided users' ratings predictions against the
	 * mean absolute error (MAE) metric.
	 */
	public static TestResult evaluateMAE(double[][] expectations, double[][] predictions) {
		int numRatings = 0;
		double sumError = 0;
		double sumSqError = 0;
		
		for (int i=0; i<expectations.length; i++) {
			for (int j=0; j<expectations[i].length; j++) {
				if (expectations[i][j] == 0.0 /*|| HybridRbmCfModelTest.ratingsPerItem[j] <= 20*/) {
					continue;
				}
				
				numRatings++;
				int expected = (int) Math.round(expectations[i][j]);
				int predicted = (int) Math.round(predictions[i][j]);
				int error = Math.abs(expected - predicted);
				sumError += error;
				sumSqError += error*error;
			}
		}
		
		TestResult result = new TestResult();
		result.MAE = (double) sumError / numRatings;
		result.RMSE = Math.sqrt((double) sumSqError / numRatings);
		result.numRatings = numRatings;
		result.numResults = 1;
			
		//System.out.println("Number of ratings to predict: " + numRatings);
		DecimalFormat df = new DecimalFormat("#.#####");
		System.out.println("MAE: " + df.format(result.MAE));
		//System.out.println("\nRMSE: " + df.format(result.RMSE));
		
		return result;
	}
	
	/**
	 * Evaluates the provided users' ratings predictions against the
	 * mean absolute error (MAE) metric.
	 */
	public static TestResult evaluateMAE(double[][] expectations, double[][] predictions, int userIndex) {
		int numRatings = 0;
		double sumError = 0;
		double sumSqError = 0;
		
		for (int i=0; i<=userIndex; i++) {
			for (int j=0; j<expectations[i].length; j++) {
				if (expectations[i][j] == 0.0) {
					continue;
				}
				
				numRatings++;
				int expected = (int) Math.round(expectations[i][j]);
				int predicted = (int) Math.round(predictions[i][j]);
				int error = Math.abs(expected - predicted);
				sumError += error;
				sumSqError += error*error;
			}
		}
		
		TestResult result = new TestResult();
		result.MAE = (double) sumError / numRatings;
		result.RMSE = Math.sqrt((double) sumSqError / numRatings);
		result.numRatings = numRatings;
		result.numResults = 1;
			
		//System.out.println("Number of ratings to predict: " + numRatings);
		DecimalFormat df = new DecimalFormat("#.#####");
		System.out.println("\nMAE: " + df.format(result.MAE));
		System.out.println("\nRMSE: " + df.format(result.RMSE));
		
		return result;
	}
	
	/**
	 * Evaluates the provided users' ratings predictions against the
	 * root mean squared error (RMSE) metric.
	 */
	public static void evaluateRMSE(double[][] expectations, double[][] predictions) {
		int numRatings = 0;
		double sqError = 0.0;
		
		for (int i=0; i<expectations.length; i++) {
			for (int j=0; j<expectations[i].length; j++) {
				if (expectations[i][j] == 0.0) {
					continue;
				}
				
				numRatings++;
				int expected = (int) Math.round(expectations[i][j]);
				int predicted = (int) Math.round(predictions[i][j]);
				
				sqError += (expected - predicted)*(expected - predicted);
			}
		}
		
		System.out.println("Ratings count: " + numRatings);
		DecimalFormat df = new DecimalFormat("#.#####");
		System.out.println("RMSE: " + df.format(Math.sqrt((double) sqError / numRatings)));
	}
	
	/**
	 * Calculates the standard deviation of the mean absolute error (MAE) evaluated for
	 * the provided users' ratings predictions.
	 */
	public static double calculateSdMAE(double MAE, double[][] expectations, double[][] predictions) {
		int numRatings = 0;
		double sumSq = 0;
		
		for (int i=0; i<expectations.length; i++) {
			for (int j=0; j<expectations[i].length; j++) {
				if (expectations[i][j] == 0.0) {
					continue;
				}
				
				numRatings++;
				int expected = (int) Math.round(expectations[i][j]);
				int predicted = (int) Math.round(predictions[i][j]);
				
				sumSq += Math.pow((Math.abs(expected - predicted) - MAE), 2);
			}
		}
		
		double sd = Math.sqrt((double) sumSq / (numRatings - 1));
		
//		DecimalFormat df = new DecimalFormat("#.#####");
//		System.out.println("MAE standard deviation: " + df.format(sd));
		
		return sd;
	}
	
	private static TestResult evaluatePRAF(int truePos, int falsePos, int trueNeg, int falseNeg) {
		TestResult result = new TestResult();
		result.numResults = 1;
		result.precision = (double)truePos/(truePos + falsePos);
		result.recall = (double)truePos/(truePos + falseNeg);
		result.accuracy = (double)(truePos+trueNeg)/(truePos + trueNeg + falsePos + falseNeg);
		
		DecimalFormat df = new DecimalFormat("#.#####");
		System.out.println("Precision: " + df.format(result.precision));
		System.out.println("Recall: " + df.format(result.recall));
		System.out.println("Accuracy: " + df.format(result.accuracy));
		System.out.println("F-measure: " + df.format((2 * result.precision * result.recall) / (result.precision + result.recall)));
		
		return result;
	}
	
	/**
	 * Prints the confidence interval around the provided mean absolute error (MAE) metric.
	 */
	private static void printConfidenceInterval(TestResult result) {
		printConfidenceInterval(result.MAE, result.sdMAE, result.numResults * result.numRatings);
	}
	
	private static void printConfidenceInterval(double MAE, double sdMAE, int n) {
		double hw = 1.96 * sdMAE / Math.sqrt(n);
		
		DecimalFormat df = new DecimalFormat("#.#####");
		System.out.println("MAE Confidence Interval: [" + df.format(MAE - hw) + ", " + df.format(MAE + hw) + "]");
	}
}
