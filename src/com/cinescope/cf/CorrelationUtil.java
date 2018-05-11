package com.cinescope.cf;

public class CorrelationUtil {
	/**
	 * Applies neighborhood-based collaborative filtering using Pearson correlation
	 * for calculating users similarity.
	 * 
	 * <p>This is an original modification to the classical Pearson-based algorithm proposed
	 * in this master thesis that works with both original user-item matrix data and the
	 * imputed user-item matrix produced by another model such as the RBM CF model.
	 */
	public static double[][] applyPearsonFiltering(double[][] originalMatrix, double[][] imputedMatrix) {
		double[][] result = new double[originalMatrix.length][originalMatrix[0].length];
		
		System.out.println("--- Calculating averages ---");
		// Calculate averages form the original data only
		double[] itemAverages = calculateItemAverages(originalMatrix);
		System.out.println("--- Calculating correlations ---");
		// Calculate correlations based on both the original and the imputed data
		double[][] itemCorrelations = calcualteItemCorrelations(imputedMatrix, itemAverages, originalMatrix);
		
		System.out.println("--- Calculating predictions ---");
		for (int item = 0; item < originalMatrix[0].length; item++) {
			for (int user = 0; user < originalMatrix.length; user++) {
				if (originalMatrix[user][item] == 0.0) {
					double sum1 = 0.0;
					double sum2 = 0.0;
					
					for (int otherItem = 0; otherItem < originalMatrix[0].length; otherItem++) {
						if (originalMatrix[user][otherItem] > 0) {
							sum1 += (imputedMatrix[user][otherItem] - itemAverages[otherItem]) *
									itemCorrelations[item][otherItem];
							sum2 += Math.abs(itemCorrelations[item][otherItem]);
						}
					}
					result[user][item] = itemAverages[item] + sum1 / sum2;
				}
			}
		}
		
		return result;
	}
	
	public static double[][] calcualteItemCorrelations(double[][] userItemMatrix) {
		// Calculate averages form the original data only
		double[] itemAverages = calculateItemAverages(userItemMatrix);
		// Calculate correlations based on both the original and the imputed data
		double[][] itemCorrelations = calcualteItemCorrelations(userItemMatrix, itemAverages, null);
		
		return itemCorrelations;
	}
	
	private static double[][] calcualteItemCorrelations(double[][] userItemMatrix, double[] itemAverages, double[][] originalMatrix) {
		double[][] result = new double[userItemMatrix[0].length][userItemMatrix[0].length];
		
		for (int item1 = 0; item1 < userItemMatrix[0].length - 1; item1++) {
			//System.out.println("--- Calculating Item: " + item1);
			for (int item2 = item1 + 1; item2 < userItemMatrix[0].length; item2++) {
				double sum = 0.0;
				double item1Sum = 0.0;
				double item2Sum = 0.0;
				int corated = 0;
				
				for (int user = 0; user < userItemMatrix.length; user++) {
					if (originalMatrix != null && originalMatrix[user][item1] > 0 && originalMatrix[user][item2] > 0) {
						corated++;
					}
					if (userItemMatrix[user][item1] > 0 && userItemMatrix[user][item2] > 0) {
						double item1Diff = (userItemMatrix[user][item1] - itemAverages[item1]);
						double item2Diff = (userItemMatrix[user][item2] - itemAverages[item2]);
						sum += item1Diff * item2Diff;
						item1Sum += item1Diff * item1Diff;
						item2Sum += item2Diff * item2Diff;
					}
				}
				
				//System.out.println("Co-rated: " + corated);
				
				double pc = sum / (Math.sqrt(item1Sum) * Math.sqrt(item2Sum));
				if (corated < 50) {
					pc = pc * ((double)corated / 50.0);
				}
				result[item1][item2] = pc;
				result[item2][item1] = pc;
			}
		}
		
		
		return result;
	}
	
	private static double[] calculateItemAverages(double[][] userItemMatrix) {
		double[] result = new double[userItemMatrix[0].length];
		
		for (int item = 0; item < userItemMatrix[0].length; item++) {
			double sum = 0.0;
			int count = 0;
			
			for (int user = 0; user < userItemMatrix.length; user++) {
				if (userItemMatrix[user][item] > 0) {
					count++;
					sum += userItemMatrix[user][item];
				}
			}
			
			result[item] = sum / count;
		}
		
		return result;
	}
	
	public static double[][] calculateUserCorrelations(double[][] userItemMatrix) {
		double[] userAverages = calculateUserAverages(userItemMatrix);
		return calculateUserCorrelations(userItemMatrix, userAverages);
	}
	
	private static double[][] calculateUserCorrelations(double[][] userItemMatrix, double[] userAverages) {
		double[][] result = new double[userItemMatrix.length][userItemMatrix.length];
		
		for (int user1 = 0; user1 < userItemMatrix.length - 1; user1++) {
			for (int user2 = user1 + 1; user2 < userItemMatrix.length; user2++) {
				double sum = 0.0;
				double user1Sum = 0.0;
				double user2Sum = 0.0;
				
				for (int item = 0; item < userItemMatrix[0].length; item++) {
					if (userItemMatrix[user1][item] > 0 && userItemMatrix[user2][item] > 0) {
						double user1Diff = (userItemMatrix[user1][item] - userAverages[user1]);
						double user2Diff = (userItemMatrix[user2][item] - userAverages[user2]);
						sum += user1Diff * user2Diff;
						user1Sum += user1Diff * user1Diff;
						user2Sum += user2Diff * user2Diff;
					}
				}
				
				double pc = sum / (Math.sqrt(user1Sum) * Math.sqrt(user2Sum));
				result[user1][user2] = pc;
				result[user2][user1] = pc;
			}
		}
		
		
		return result;
	}
	
	private static double[] calculateUserAverages(double[][] userItemMatrix) {
		double[] result = new double[userItemMatrix.length];
		
		for (int user = 0; user < userItemMatrix.length; user++) {
			double sum = 0.0;
			int count = 0;
			
			for (int item = 0; item < userItemMatrix[user].length; item++) {
				if (userItemMatrix[user][item] > 0) {
					count++;
					sum += userItemMatrix[user][item];
				}
			}
			
			result[user] = sum / count;
		}
		
		return result;
	}
}
