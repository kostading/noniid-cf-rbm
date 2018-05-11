package com.cinescope.cf;

/**
 * Contains various utilities for workign with user-rating matrices.
 */
public final class MatrixUtil {
	public static double[][] imputeMissingValues(double[][] matrix) {
		double[][] result = new double[matrix.length][matrix[0].length];
		
		double[] columnAverages = new double[matrix[0].length];
		for (int column=0; column<matrix[0].length; column++) {
			columnAverages[column] = getColumnAverage(matrix, column);
		}
		
		for (int i=0; i<matrix.length; i++) {
			for (int j=0; j<matrix[i].length; j++) {
				if (matrix[i][j] == 0.0) {
					result[i][j] = columnAverages[j];
				} else {
					result[i][j] = matrix[i][j];
				}
			}
		}
		
		return result;
	}

	public static double[][] toBinaryMatrix(double[][] userItemMatrix) {
		double[][] result = new double[userItemMatrix.length][userItemMatrix[0].length * 5];
		
		for (int i=0; i<userItemMatrix.length; i++) {
			for (int j=0; j<userItemMatrix[i].length; j++) {
				if (userItemMatrix[i][j] != 0) {
					result[i][j*5 + (int)userItemMatrix[i][j] - 1] = 1;
				}
			}
		}
		
		return result;
	}

	public static double[][] toRealMatrix(double[][] userItemMatrix) {
		double[][] result = new double[userItemMatrix.length][userItemMatrix[0].length / 5];
		
		for (int i=0; i<userItemMatrix.length; i++) {
			for (int j=0; j<userItemMatrix[i].length; j+=5) {
				for (int k=0; k<5; k++) {
					if (userItemMatrix[i][j+k] != 0) {
						result[i][j/5] = k + 1;
						break;
					}
				}
			}
		}
		
		return result;
	}
	
	public static double getColumnAverage(double[][] matrix, int column) {
		double sum = 0.0;
		
		for (int i=0; i<matrix.length; i++) {
			sum += matrix[i][column];
		}
		
		return sum / matrix.length;
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
	
	private static double[][] transposeBinaryMatrix(double[][] original) {
		double[][] transposed = new double[original[0].length/5][original.length*5];
		
		for (int i=0; i<original.length; i++) {
			for (int j=0; j<original[i].length; j+=5) {
				for (int k=0; k<5; k++) {
					transposed[j/5][i*5+k] = original[i][j+k];
				}
			}
		}
		
		return transposed;
	}
}
