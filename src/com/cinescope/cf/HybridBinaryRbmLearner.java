package com.cinescope.cf;

import com.syvys.jaRBM.RBM;
import com.syvys.jaRBM.Math.Matrix;

/**
 * A modification to the Contrastive Divergence (CD) RBM learner that works with sparse
 * data i.e. user's vectors having missing ratings. Note that the code is copied from jaRBM
 * original implementation and modified to handle the missing ratings properly in order
 * to make the network able to predict them.
 */
public class HybridBinaryRbmLearner {
    
    /**
     * Creates a new instance of CDStochasticRBMLearner
     */
    public HybridBinaryRbmLearner() {
    }
    
    public static double Learn(RBM userBasedRbm, RBM itemBasedRbm, double[][] data, double[][] transposedData) {
    	// user-based
        double[][] userHiddenActivities = userBasedRbm.getHiddenActivitiesFromVisibleData(data);
        double[][] userHiddenData = userBasedRbm.GenerateHiddenUnits(userHiddenActivities);
        double[][] userSum  = userBasedRbm.getDownwardSWSum(userHiddenData);
        // item-based
        double[][] itemHiddenActivities = itemBasedRbm.getHiddenActivitiesFromVisibleData(transposedData);
        double[][] itemHiddenData = itemBasedRbm.GenerateHiddenUnits(itemHiddenActivities);
        double[][] itemSum  = itemBasedRbm.getDownwardSWSum(itemHiddenData);

        for (int i=0; i<userSum.length; i++) {
        	for (int j=0; j<userSum[i].length/5; j++) {
        		for (int k=0; k<5; k++) {
        			double newVal = (userSum[i][j*5+k] + itemSum[j][i*5+k]) / 2.0;
        			userSum[i][j*5+k] = newVal;
        			itemSum[j][i*5+k] = newVal;
        		}
        	}
        }
        double[][] userNegPhaseVisible = userBasedRbm.getVisibleLayer().getActivationProbabilities(userSum);
        double[][] itemNegPhaseVisible = itemBasedRbm.getVisibleLayer().getActivationProbabilities(itemSum);
        
        // null values missing in original data
        RbmBasedCF.nullMissingValuesBinary(data, userNegPhaseVisible);
        // null values missing in original data
        RbmBasedCF.nullMissingValuesBinary(transposedData, itemNegPhaseVisible); 
        
        double[][] userNegPhaseHidden = userBasedRbm.getHiddenActivitiesFromVisibleData(userNegPhaseVisible);
        double[][] itemNegPhaseHidden = itemBasedRbm.getHiddenActivitiesFromVisibleData(itemNegPhaseVisible);
        
        // update weights for user-based
        double[][] userWeightUpdates = getConnectionWeightUpdates(userBasedRbm, data, userHiddenActivities, userNegPhaseVisible, userNegPhaseHidden);
        updateWeights(userBasedRbm, userWeightUpdates);
        userBasedRbm.UpdateHiddenBiases(userHiddenActivities, userNegPhaseHidden);
        userBasedRbm.UpdateVisibleBiases(data, userNegPhaseVisible);
        
        // update weights for item-based
        double[][] itemWeightUpdates = getConnectionWeightUpdates(itemBasedRbm, transposedData, itemHiddenActivities, itemNegPhaseVisible, itemNegPhaseHidden);
        updateWeights(itemBasedRbm, itemWeightUpdates);
        itemBasedRbm.UpdateHiddenBiases(itemHiddenActivities, itemNegPhaseHidden);
        itemBasedRbm.UpdateVisibleBiases(transposedData, itemNegPhaseVisible);
        
//        userHiddenActivities = userBasedRbm.getHiddenActivitiesFromVisibleData(data);
//        userSum  = userBasedRbm.getDownwardSWSum(userHiddenActivities);
//        
//        itemHiddenActivities = itemBasedRbm.getHiddenActivitiesFromVisibleData(transposedData);
//        itemSum  = itemBasedRbm.getDownwardSWSum(itemHiddenActivities);
//        
//        for (int i=0; i<userSum.length; i++) {
//        	for (int j=0; j<userSum[i].length/5; j++) {
//        		for (int k=0; k<5; k++) {
//        			userSum[i][j*5+k] = (userSum[i][j*5+k] + itemSum[j][i*5+k]) / 2.0;
//        		}
//        	}
//        }
//        
//        userNegPhaseVisible = userBasedRbm.getVisibleLayer().getActivationProbabilities(userSum);
//        // null values missing in original data
//        RbmBasedCF.nullMissingValuesBinary(data, userNegPhaseVisible);
//        double[][] visibleData = userBasedRbm.GenerateVisibleUnits(userNegPhaseVisible);
//        
//        return Matrix.getMeanSquaredError(data, visibleData);
        return 0;
    }
    
    public static double[][] getConnectionWeightUpdates(RBM rbm, double[][] data, double[][] hiddenActivities, 
                                double[][] generatedData, double[][] negativePhaseHiddenActivities) {
        int numVisibleUnits = rbm.getNumVisibleUnits();
        int numHiddenUnits = rbm.getNumHiddenUnits();
        //////////////////////////////////////////////////////////////
        //          Updating Symmetric Weights
        //////////////////////////////////////////////////////////////
        double[][] weightUpdates = new double[numVisibleUnits][numHiddenUnits];
        
        final double[][] positivePhaseProduct = rbm.getVisibleHiddenStateProducts(data, hiddenActivities);
        final double[][] negativePhaseProduct = rbm.getVisibleHiddenStateProducts(generatedData, negativePhaseHiddenActivities);
        
        final double[][] previousWeightIncrement = rbm.getConnectionWeightIncrement();
        final double[][] weightConnections = rbm.getConnectionWeights();
        final double momentum = rbm.getMomentum();
        final double learningRate = rbm.getLearningRate();
        final double weightCost = rbm.getWeightCost();
        for (int v = 0; v < numVisibleUnits; v++) {
            for (int h = 0; h < numHiddenUnits; h++) {
                // Calculating how much to change each symmetric weight by.
                weightUpdates[v][h] = momentum * previousWeightIncrement[v][h] +
                                      learningRate * ( (positivePhaseProduct[v][h] - negativePhaseProduct[v][h]) / data.length -
                                      weightCost * weightConnections[v][h]);
            }
        }
        return weightUpdates;
    }
    
    public static double[] getConnectionWeightUpdatesVector(RBM rbm, double[][] data, double[][] hiddenActivities, 
                                double[][] generatedData, double[][] negativePhaseHiddenActivities) {
        // Updating Weights
        double[] weightUpdates = new double[rbm.getConnectionWeightVectorLength()];
        
        final double[] positivePhaseProduct = rbm.getVisibleHiddenStateProductsVector(data, hiddenActivities);
        final double[] negativePhaseProduct = rbm.getVisibleHiddenStateProductsVector(generatedData, negativePhaseHiddenActivities);
        
        final double[] previousWeightIncrement = rbm.getConnectionWeightIncrementVector();
        final double[] weightConnections = rbm.getConnectionWeightVector();
        final double momentum = rbm.getMomentum();
        final double learningRate = rbm.getLearningRate();
        final double weightCost = rbm.getWeightCost();
        for (int w = 0; w < positivePhaseProduct.length; w++) {
                // Calculating how much to change each symmetric weight by.
                weightUpdates[w] = momentum * previousWeightIncrement[w] +
                                      learningRate * ( (positivePhaseProduct[w] - negativePhaseProduct[w]) / data.length -
                                      weightCost * weightConnections[w]);
        }
        return weightUpdates;
    }
    
    public static void updateWeights (RBM rbm, double[] weightUpdatesVector) {
        //////////////////////////////////////////////////////////////
        //          Updating Symmetric Weights
        //////////////////////////////////////////////////////////////
        double[] newWeights = rbm.getConnectionWeightVector();
        for (int w = 0; w < newWeights.length; w++) {
            newWeights[w] += weightUpdatesVector[w];
        }
        rbm.setConnectionWeightVector(newWeights);
        rbm.setConnectionWeightIncrementVector(weightUpdatesVector);
    }
    public static void updateWeights(RBM rbm, double[][] weightUpdates) {
        double[][] weights = rbm.getConnectionWeights();
        for (int v = 0; v < weights.length; v++) {
            for (int h = 0; h < weights[0].length; h++) {
                weights[v][h] += weightUpdates[v][h];
            }
        }
        rbm.setConnectionWeightIncrement(weightUpdates);
    }
    
}
