package com.cinescope.cf.test;

import java.io.File;
import java.io.IOException;

import com.cinescope.cf.TestResult;
import com.cinescope.data.MovieLensDataReader;

/**
 * Conducts experiments with various collaborative filtering (CF) models.
 */
public class CfModelTestExecutor {
	public static int USER_HIDDEN_SIZE = 10;
	public static int ITEM_HIDDEN_SIZE = 10;

	public static void main(String[] args) throws IOException {
		File movieLensDir;
		
		if (args == null || args.length == 0) {
			System.out.println("The MovieLens data set directory was not specified. " +
					"So looking for it in the current directory: ");
			movieLensDir = new File(".", Config.ML_DATA_DIR_NAME);
			System.out.println(movieLensDir.getCanonicalPath());
		} else {
			movieLensDir = new File(args[0]);
			
			if (!movieLensDir.exists()) {
				throw new IllegalArgumentException(
						"The specified MovieLens data set directory does not exist:" + movieLensDir);
			}
		}
		
		testCfModels(movieLensDir);
	}
	
	/**
	 * Runs the tests of various collaborative filtering (CF) models against the MovieLens
	 * data set located in the provided directory.
	 */
	private static void testCfModels(File movieLensDir) {
//		CfModelTest rbmCfModelTest = new RbmCfModelTest();
//		testCfModelWithCV(rbmCfModelTest, movieLensDir);
		
//		CfModelTest hybridBinaryRbmCfModelTest = new HybridBinaryRbmCfModelTest();
//		testCfModelWithCVBig(hybridBinaryRbmCfModelTest, movieLensDir);
		
//		CfModelTest hybridRbmCfModelTest = new HybridRbmCfModelTest();
//		testCfModelWithCV(hybridRbmCfModelTest, movieLensDir);
		
		CfModelTest hybridRbmCfModelTest = new HybridRbmCfModelTest();
		testCfModelWithCVBig(hybridRbmCfModelTest, movieLensDir);
		
//		CfModelTest combinedHybridRbmCfModelTest = new CombinedHybridRbmCfModelTest();
//		testCfModelWithCVBig(combinedHybridRbmCfModelTest, movieLensDir);
		
//		CfModelTest hybridRbmCfModelTest2 = new HybridRbmCfModelTest2();
//		testCfModelWithCV(hybridRbmCfModelTest2, movieLensDir);
		
//		CfModelTest rbmPearsonCfModelTest = new RbmPearsonCfModelTest();
//		testCfModelWithCVBig(rbmPearsonCfModelTest, movieLensDir);
		
//		CfModelTest combinedCfModelTest = new CombinedCfModelTest();
//		testCfModelWithCVBig(combinedCfModelTest, movieLensDir);
		
//		CfModelTest multipleRbmCfTest = new MultipleRbmCfModelTest();
//		testCfModelWithCVBig(multipleRbmCfTest, movieLensDir);
		
//		CfModelTest clusterRbmCfModelTest = new ClusterRbmCfModelTest();
//		testCfModelWithCVBig(clusterRbmCfModelTest, movieLensDir);
	}
	
	/**
	 * Tests the provided collaborative filtering (CF) model on the provided MovieLens
	 * data set while performing 5-fold cross-validation (CV).
	 */
	private static void testCfModelWithCV(CfModelTest cfModelTest, File movieLensDir) {
		System.out.println("\nTesting model: " + cfModelTest.toString());
		File infoFile = new File(movieLensDir, Config.INFO_FILE_NAME);
		TestResult bestResult = null;
		
		for (USER_HIDDEN_SIZE=40; USER_HIDDEN_SIZE<=40; USER_HIDDEN_SIZE+=10) {
			for (ITEM_HIDDEN_SIZE=USER_HIDDEN_SIZE; ITEM_HIDDEN_SIZE<=USER_HIDDEN_SIZE; ITEM_HIDDEN_SIZE+=10) {
				System.out.println("\nUser hidden size: " + USER_HIDDEN_SIZE);
				System.out.println("Item hidden size: " + ITEM_HIDDEN_SIZE);
				TestResult testResult = new TestResult();
				// Cross-validation
				for (int i=1; i<=5; i++) {
					String trainFileName = String.format(Config.TRAIN_FILE_PATTERN, i);
					String testFileName = String.format(Config.TEST_FILE_PATTERN, i);
					
					File trainFile = new File(movieLensDir, trainFileName);
					File testFile = new File(movieLensDir, testFileName);
					
					TestResult currentResult = testCfModel(cfModelTest, infoFile, trainFile, testFile);
					testResult.accumulateResult(currentResult);
				}
				
				testResult.printTotals();
				
				if (bestResult == null || bestResult.MAE > testResult.MAE) {
					bestResult = testResult;
				}
			}
		}
		
		System.out.println("\nCross-validation results on model: " + cfModelTest.toString());
		bestResult.printTotals();
	}
	
	private static TestResult testCfModel(CfModelTest cfModelTest,
										  File infoFile,
										  File trainFile,
										  File testFile) {
		MovieLensDataReader dataReader = new MovieLensDataReader(infoFile);
		
		double[][] trainMatrix = dataReader.readUserItemMatrix(trainFile);
		double[][] testMatrix = dataReader.readUserItemMatrix(testFile);
		
		try {
			System.out.println("\nTraining on: " + trainFile.getCanonicalPath());
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		TestResult result = cfModelTest.execute(trainMatrix, testMatrix);
		
		return result;
	}
	
	/**
	 * Tests the provided collaborative filtering (CF) model on the provided MovieLens
	 * data set while performing 5-fold cross-validation (CV).
	 */
	private static void testCfModelWithCVBig(CfModelTest cfModelTest, File movieLensDir) {
		System.out.println("\nTesting model: " + cfModelTest.toString());
		File infoFile = new File(movieLensDir, Config.INFO_FILE_NAME);
		TestResult bestResult = null;
		
		for (USER_HIDDEN_SIZE=70; USER_HIDDEN_SIZE<=70; USER_HIDDEN_SIZE+=10) {
			for (ITEM_HIDDEN_SIZE=USER_HIDDEN_SIZE; ITEM_HIDDEN_SIZE<=USER_HIDDEN_SIZE; ITEM_HIDDEN_SIZE+=10) {
				System.out.println("\nUser hidden size: " + USER_HIDDEN_SIZE);
				System.out.println("Item hidden size: " + ITEM_HIDDEN_SIZE);
				
				TestResult testResult = new TestResult();
				// Cross-validation
				for (int i=2; i<3; i++) {		
					File ratingsFile = new File(movieLensDir, "ratings.dat");
					
					MovieLensDataReader dataReader = new MovieLensDataReader(infoFile);
					double[][] trainMatrix = dataReader.readUserItemMatrixTrain(ratingsFile, i);
					double[][] testMatrix = dataReader.readUserItemMatrixTest(ratingsFile, i);
					
					try {
						System.out.println("\nTraining on: " + ratingsFile.getCanonicalPath() + i);
					} catch (IOException e) {
						e.printStackTrace();
					}
					
					TestResult currentResult = cfModelTest.execute(trainMatrix, testMatrix);
					testResult.accumulateResult(currentResult);
				}
				
				testResult.printTotals();
				
	//			if (bestResult == null || bestResult.MAE > testResult.MAE) {
	//				bestResult = testResult;
	//			}
			}
		}
		
//		System.out.println("\nCross-validation results on model: " + cfModelTest.toString());
//		bestResult.printTotals();
	}

}
