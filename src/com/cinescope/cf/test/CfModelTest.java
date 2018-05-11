package com.cinescope.cf.test;

import com.cinescope.cf.TestResult;

/**
 * Defines the interface that all collaborative filtering (CF) model test procedures
 * should implement.
 */
public interface CfModelTest {
	/**
	 * The entry point of the particular collaborative filtering (CF) model test. 
	 */
	public TestResult execute(double[][] trainMatrix, double[][] testMatrix);
}
