package com.cinescope.cf;

import java.text.DecimalFormat;

import com.syvys.jaRBM.RBM;
import com.syvys.jaRBM.RBMImpl;
import com.syvys.jaRBM.RBMNet;
import com.syvys.jaRBM.IO.BatchDatasourceReaderImpl;
import com.syvys.jaRBM.Layers.Layer;
import com.syvys.jaRBM.Layers.LinearLayer;
import com.syvys.jaRBM.Layers.LogisticLayer;
import com.syvys.jaRBM.Layers.StochasticBinaryLayer;
import com.syvys.jaRBM.RBMNetLearn.RBMNetLearner;

/**
 * Contains models for collaborative filtering (CF) based on Restricted Boltzmann
 * Machines (RBM's).
 */
public final class RbmBasedCF {
	/**
	 * Modifies all values in the provided <code>currentData</code> to be equal to zero
	 * if the values were originally equal to zero.
	 * 
	 * <p>This helps making our modified contrastive divergence learning procedure being
	 * able to train a model that generates predictions for the missing user's ratings
	 * values.
	 */
    public static void nullMissingValues(double[][] originalData, double[][] currentData) {
    	for (int i=0; i<currentData.length; i++) {
    		for (int j=0; j<currentData[i].length; j++) {
    			if (originalData[i][j] == 0.0) {
    				currentData[i][j] = 0.0;
    			}
    		}
    	}
    }
    
    public static void nullMissingValuesBinary(double[][] originalData, double[][] currentData) {
    	for (int i=0; i<currentData.length; i++) {
    		for (int j=0; j<currentData[i].length; j+=5) {
    			boolean shouldNull = true;
				for (int k=0; k<5; k++) {
					if (originalData[i][j+k] != 0) {
						shouldNull = false;
						break;
					}
				}
				if (shouldNull) {
					for (int k=0; k<5; k++) {
						currentData[i][j+k] = 0;
					}
				}
    		}
    	}
    }
	
    /**
     * Learns an RBM-based model that is able to model real-valued user's ratings
     * directly with its visible layer and also generate predictions for the missing
     * user's ratings as proposed in the master thesis.
     */
	public static RBM learnRBM(double[][] userItemMatrix, RBM rbm, int hiddenUnits, double[][] testMatrix) {
        int visibleUnits = userItemMatrix[0].length;
   
//        Layer visibleLayer = new CfSoftmaxLayer(visibleUnits);
        Layer visibleLayer = new LinearLayer(visibleUnits);
        
        Layer hiddenLayer = new LogisticLayer(hiddenUnits);
//      Layer hiddenLayer = new StochasticBinaryLayer(hiddenUnits);
        
        if (rbm == null) {
        	rbm = new RBMImpl(visibleLayer, hiddenLayer);
        }
        
        for (int epochs = 1; epochs <= 1000; epochs++) {
            double error = SparseCDStochasticRBMLearner.Learn(rbm, userItemMatrix);
//        	DecimalFormat df = new DecimalFormat("#.#####");
//        	System.out.println("Epoch: " + epochs + "; Reconstruction error: " + df.format(error));
            if (epochs % 50 == 0) {
            	//System.out.println("Evaluation results on epoch: " + epochs);
	            double[][] hiddenActivities = rbm.getHiddenActivitiesFromVisibleData(userItemMatrix);
	            double[][] hiddenData = rbm.GenerateHiddenUnits(hiddenActivities);
	            hiddenActivities = null;
	            // generation phase
	            double[][] visibleActivities = rbm.getVisibleActivitiesFromHiddenData(hiddenData);
	            hiddenData = null;
	            double[][] visibleData = rbm.GenerateVisibleUnits(visibleActivities);
	            visibleActivities = null;
	            
//	            visibleData = MatrixUtil.toRealMatrix(visibleData);
	            EvalUtil.evaluatePredictions(testMatrix, visibleData);
            }
        }
        System.out.println();
        
        return rbm;
	}
	
	public static RBM batchLearnRBM(double[][] userItemMatrix, RBM rbm) {
        int visibleUnits = userItemMatrix[0].length;
        
        Layer visibleLayer = new LinearLayer(visibleUnits);
        
        int hiddenUnits = 50;
        Layer hiddenLayer = new LogisticLayer(hiddenUnits);
        //Layer hiddenLayer = new StochasticBinaryLayer(hiddenUnits);
        
        if (rbm == null) {
        	rbm = new RBMImpl(visibleLayer, hiddenLayer);
        }
        
        for (int epochs = 1; epochs <= 5; epochs++) {
        	BatchDatasourceReaderImpl batchReader = new BatchDatasourceReaderImpl(userItemMatrix);
        	while ( batchReader.hasNext() ) {
                double[][] nextVector = batchReader.getNext(500);
                double error = SparseCDStochasticRBMLearner.Learn(rbm, nextVector);
            	DecimalFormat df = new DecimalFormat("#.#####");
            	System.out.println("Epoch " + epochs + ": " + df.format(error));
        	}
        }
        System.out.println();
        
        return rbm;
	}
	
    /**
     * Learns a deep belief network (DBN) model that is able to model real-valued
     * user's ratings and also generate predictions for the missing ratings.
     */
	public static RBMNet learnDBN(double[][] userItemMatrix) {        
        int visibleUnits = userItemMatrix[0].length;
        Layer visibleLayer = new LinearLayer(visibleUnits);
        
        int hiddenUnits = 50;
        Layer hiddenLayer = new LogisticLayer(hiddenUnits);
	        
	    RBMImpl rbm = new RBMImpl(visibleLayer, hiddenLayer);
	    RBMNet rbmNet = new RBMNet(rbm);
	    
        RBMImpl rbm2 = new RBMImpl(new LogisticLayer(rbmNet.getNumOfTopHiddenUnits()), new LogisticLayer(800));
        try {
			rbmNet.AddRBM(rbm2);
		} catch (Exception e) {
			e.printStackTrace();
		}
        
	    BatchDatasourceReaderImpl batchReader = new BatchDatasourceReaderImpl(userItemMatrix);
	        
	    SparseGreedyLearner greedyLearner = new SparseGreedyLearner(rbmNet, batchReader, 1000);
	    System.out.println("\nGreedy training error = " + greedyLearner.Learn(300));
	    rbmNet = greedyLearner.getLearnedRBMNet();
	    
	    return rbmNet;
	}
	
    /**
     * Learns an auto-encoder model that is able to model real-valued user's ratings
     * and also generate predictions for the missing ratings.
     */
	public static RBMNet learnAutoencoder(double[][] userItemMatrix) {
        int visibleUnits = userItemMatrix[0].length;
        Layer visibleLayer = new LinearLayer(visibleUnits);
        
        int hiddenUnits = 50;
        Layer hiddenLayer = new LogisticLayer(hiddenUnits);
	        
	    RBMImpl rbm = new RBMImpl(visibleLayer, hiddenLayer);
	    RBMNet rbmNet = new RBMNet(rbm);
	    
	    RBMImpl rbm2 = new RBMImpl(new LogisticLayer(rbmNet.getNumOfTopHiddenUnits()), new StochasticBinaryLayer(2));
        try {
			rbmNet.AddRBM(rbm2);
		} catch (Exception e) {
			e.printStackTrace();
		}
        
	    BatchDatasourceReaderImpl batchReader = new BatchDatasourceReaderImpl(userItemMatrix);
	        
	    SparseGreedyLearner greedyLearner = new SparseGreedyLearner(rbmNet, batchReader, 1000);
	    System.out.println("\nGreedy training error = " + greedyLearner.Learn(300));
	    rbmNet = greedyLearner.getLearnedRBMNet();
	    
	    RBMNetLearner autoencoderLearner = new SparseAutoencoderLearner(rbmNet, batchReader, 1000);
        for (int i = 0; i < 100; i++) {
            double error = autoencoderLearner.Learn();
            System.out.println("Backpropagation training error epoch: "+i+" = " + error);
        } 
        rbmNet = autoencoderLearner.getLearnedRBMNet().clone();
        
        return rbmNet;
	}

}
