package com.cinescope.cf;

/**
 * Contains various memory-based algorithms for collaborative filtering (CF) such as
 * Pearson correlation-based filtering and others.
 */
public final class MemoryBasedCF {
	/**
	 * Applies neighborhood-based collaborative filtering using cosine-based distance
	 * for calculating users similarity.
	 */
	public static double[][] applyUserCosineFiltering(double[][] originalMatrix, double[][] imputedMatrix) {
		double[][] result = new double[originalMatrix.length][originalMatrix[0].length];
		
		double[] userAverages = calculateUserAverages(originalMatrix);
		double[][] userCorrelations = calcualteUserSimilarity(imputedMatrix);
		
		System.out.println("\nCalculating predictons");
		
		for (int user = 0; user < originalMatrix.length; user++) {
			for (int item = 0; item < originalMatrix[user].length; item++) {
				if (originalMatrix[user][item] == 0.0) {
					double sum1 = 0.0;
					double sum2 = 0.0;
					
					for (int otherUser = 0; otherUser < originalMatrix.length; otherUser++) {
						if (originalMatrix[otherUser][item] > 0) {
							sum1 += (originalMatrix[otherUser][item] - userAverages[otherUser]) *
									userCorrelations[user][otherUser];
							sum2 += Math.abs(userCorrelations[user][otherUser]);
						}
					}
					result[user][item] = userAverages[user] + sum1 / sum2;
				}
			}
		}
		
		return result;
	}
	
	public static double[][] calcualteUserSimilarity(double[][] userItemMatrix) {
		double[][] result = new double[userItemMatrix.length][userItemMatrix.length];
		
		for (int user1 = 0; user1 < userItemMatrix.length - 1; user1++) {
			for (int user2 = user1 + 1; user2 < userItemMatrix.length; user2++) {
				
				double dot = 0.0;
				double user1Norm = 0.0;
				double user2Norm =0.0;
				
				for (int item = 0; item < userItemMatrix[0].length; item++) {
					dot += userItemMatrix[user1][item] * userItemMatrix[user2][item];
					user1Norm += userItemMatrix[user1][item] * userItemMatrix[user1][item];
					user2Norm += userItemMatrix[user2][item] * userItemMatrix[user2][item];
				}
				
				double sim = dot / (Math.sqrt(user1Norm) * Math.sqrt(user2Norm));
				result[user1][user2] = sim;
				result[user2][user1] = sim;
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
	
	/**
	 * Applies neighborhood-based collaborative filtering using cosine-based distance
	 * for calculating users similarity.
	 */
	public static double[][] applyItemCosineFiltering(double[][] originalMatrix, double[][] imputedMatrix) {
		double[][] result = new double[originalMatrix.length][originalMatrix[0].length];
		
		// Calculate averages form the original data only
		double[] itemAverages = calculateItemAverages(originalMatrix);
		
		System.out.println("\nCalculating similarities");
		
		// Calculate correlations based on both the original and the imputed data
		double[][] itemCorrelations = calcualteItemSimilarity(imputedMatrix, originalMatrix);
		
		System.out.println("\nCalculating predictons");
		
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
	
	public static double[][] calcualteItemSimilarity(double[][] userItemMatrix, double[][] originalMatrix) {
		double[][] result = new double[userItemMatrix[0].length][userItemMatrix[0].length];
		
		for (int item1 = 0; item1 < userItemMatrix[0].length - 1; item1++) {
			for (int item2 = item1 + 1; item2 < userItemMatrix[0].length; item2++) {
				
				double dot = 0.0;
				double item1Norm = 0.0;
				double item2Norm =0.0;
				int corated = 0;
				
				for (int user = 0; user < userItemMatrix.length; user++) {
					if (originalMatrix != null && originalMatrix[user][item1] > 0 && originalMatrix[user][item2] > 0) {
						corated++;
					}
					dot += userItemMatrix[user][item1] * userItemMatrix[user][item2];
					item1Norm += userItemMatrix[user][item1] * userItemMatrix[user][item1];
					item2Norm += userItemMatrix[user][item2] * userItemMatrix[user][item2];
				}
				
				double sim = dot / (Math.sqrt(item1Norm) * Math.sqrt(item2Norm));
				if (corated < 50) {
					sim = sim * ((double)corated / 50.0);
				}
				result[item1][item2] = sim;
				result[item2][item1] = sim;
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

}
