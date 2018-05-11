package com.cinescope.data;

import java.io.File;
import java.io.IOException;

public class MovieLensDataReaderTest {
	private static final String INFO_FILE_NAME = "D:\\src\\datasets\\ml-data_0\\u.info";
	private static final String DATA_FILE_NAME = "D:\\src\\datasets\\ml-data_0\\u.data";

	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) {
		File infoFile = new File(INFO_FILE_NAME);
		File dataFile = new File(DATA_FILE_NAME);
		
		MovieLensDataReader dataReader = new MovieLensDataReader(infoFile);
		double[][] userItemMatrix = dataReader.readUserItemMatrix(dataFile);
		
		System.out.println(userItemMatrix.length + ":" + userItemMatrix[0].length);
	}
}
