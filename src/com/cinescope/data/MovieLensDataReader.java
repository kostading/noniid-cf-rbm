package com.cinescope.data;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

/**
 * A convenience class responsible for reading the user-items matrix data from the
 * MovieLens data set which we're using for evaluation in the current master thesis.
 */
public final class MovieLensDataReader {
	private int _userCount;
	private int _itemCount;
	
	/**
	 * Initializes the MovieLens data set reader with the provided file containing
	 * the information about the data set - usually named <tt>u.info</tt>.
	 */
	public MovieLensDataReader(File infoFile) {
		try {
			Scanner infoScanner = new Scanner(infoFile);
			_userCount = infoScanner.nextInt();
			infoScanner.nextLine();
			_itemCount = infoScanner.nextInt();
			infoScanner.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Reads the user-item ratings matrix from the provided data file name.
	 */
	public double[][] readUserItemMatrix(File dataFile) {
		double[][] userItemMatrix = new double[_userCount][_itemCount];
		
		try {
			Scanner dataScanner = new Scanner(dataFile);
			
			while (dataScanner.hasNext()) {
				int userId = dataScanner.nextInt();
				int itemId = dataScanner.nextInt();
				int rating = dataScanner.nextInt();
				dataScanner.nextLine();
				
			    userItemMatrix[userId-1][itemId-1] = rating;
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		return userItemMatrix;
	}
	
	/**
	 * Reads the user-item ratings matrix from the provided data file name.
	 */
	public double[][] readUserItemMatrixTest(File dataFile, int offset) {
		double[][] userItemMatrix = new double[_userCount][_itemCount];
		int count = 0;
		
		try {
			Scanner dataScanner = new Scanner(dataFile);
			dataScanner.useDelimiter("::");
			int index = 0;
			
			while (dataScanner.hasNext()) {
				if (((index++ + offset) % 5) != 0) {
					dataScanner.nextLine();
					continue;
				}
				count++;
				
				int userId = dataScanner.nextInt();
				int itemId = dataScanner.nextInt();
				int rating = dataScanner.nextInt();
				dataScanner.nextLine();
				
			    userItemMatrix[userId-1][itemId-1] = rating;
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		//System.out.println("Testing ratings: " + count);
		
		return userItemMatrix;
	}
	
	/**
	 * Reads the user-item ratings matrix from the provided data file name.
	 */
	public double[][] readUserItemMatrixTrain(File dataFile, int offset) {
		double[][] userItemMatrix = new double[_userCount][_itemCount];
		int count = 0;
		
		try {
			Scanner dataScanner = new Scanner(dataFile);
			dataScanner.useDelimiter("::");
			int index = 0;
			
			while (dataScanner.hasNext()) {
				if (((index++ + offset) % 5) == 0) {
					dataScanner.nextLine();
					continue;
				}
				count++;
				
				int userId = dataScanner.nextInt();
				int itemId = dataScanner.nextInt();
				int rating = dataScanner.nextInt();
				dataScanner.nextLine();
				
			    userItemMatrix[userId-1][itemId-1] = rating;
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		//System.out.println("Training ratings: " + count);
		
		return userItemMatrix;
	}
}
