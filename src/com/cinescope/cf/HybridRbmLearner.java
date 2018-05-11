package com.cinescope.cf;

import com.syvys.jaRBM.RBM;
import com.syvys.jaRBM.Math.Matrix;

/**
 * A modification to the Contrastive Divergence (CD) RBM learner that works with sparse
 * data i.e. user's vectors having missing ratings. Note that the code is copied from jaRBM
 * original implementation and modified to handle the missing ratings properly in order
 * to make the network able to predict them.
 */
public class HybridRbmLearner {
    
    /**
     * Creates a new instance of HybridRbmLearner
     */
    public HybridRbmLearner() {
    }
    
    public static double Learn(RBM userBasedRbm, RBM itemBasedRbm, double[][] data, double[][] transposedData) {
    	// user-based
        double[][] userHiddenActivities = userBasedRbm.getHiddenActivitiesFromVisibleData(data);
        double[][] userHiddenData = userBasedRbm.GenerateHiddenUnits(userHiddenActivities);
        // item-based
        double[][] itemHiddenActivities = itemBasedRbm.getHiddenActivitiesFromVisibleData(transposedData);
        double[][] itemHiddenData = itemBasedRbm.GenerateHiddenUnits(itemHiddenActivities);
        
        // user-based negative phase
        double[][] userNegPhaseVisible = userBasedRbm.getVisibleActivitiesFromHiddenData(userHiddenData);
        // null values missing in original data
        RbmBasedCF.nullMissingValues(data, userNegPhaseVisible);
        
        // item-based negative phase
        double[][] itemNegPhaseVisible = itemBasedRbm.getVisibleActivitiesFromHiddenData(itemHiddenData);
        // null values missing in original data
        RbmBasedCF.nullMissingValues(transposedData, itemNegPhaseVisible); 
        
        double[][] negPhaseVisible = new double[userNegPhaseVisible.length][userNegPhaseVisible[0].length];
        for (int i=0; i<negPhaseVisible.length; i++) {
        	for (int j=0; j<negPhaseVisible[i].length; j++) {
        		negPhaseVisible[i][j] = (userNegPhaseVisible[i][j] + itemNegPhaseVisible[j][i]) / 2.0;
        	}
        }
        // null values missing in original data
        RbmBasedCF.nullMissingValues(data, negPhaseVisible);

        double[][] transposedNegPhaseVisible = Matrix.getTranspose(negPhaseVisible);
        double[][] userNegPhaseHidden = userBasedRbm.getHiddenActivitiesFromVisibleData(negPhaseVisible);
        double[][] itemNegPhaseHidden = itemBasedRbm.getHiddenActivitiesFromVisibleData(transposedNegPhaseVisible);
        
        // update weights for user-based
        double[][] userWeightUpdates = getConnectionWeightUpdates(userBasedRbm, data, userHiddenActivities, negPhaseVisible, userNegPhaseHidden);
        updateWeights(userBasedRbm, userWeightUpdates);
        userBasedRbm.UpdateHiddenBiases(userHiddenActivities, userNegPhaseHidden);
        userBasedRbm.UpdateVisibleBiases(data, negPhaseVisible);
        
        // update weights for item-based
        double[][] itemWeightUpdates = getConnectionWeightUpdates(itemBasedRbm, transposedData, itemHiddenActivities, transposedNegPhaseVisible, itemNegPhaseHidden);
        updateWeights(itemBasedRbm, itemWeightUpdates);
        itemBasedRbm.UpdateHiddenBiases(itemHiddenActivities, itemNegPhaseHidden);
        itemBasedRbm.UpdateVisibleBiases(transposedData, transposedNegPhaseVisible);
        
//        userHiddenActivities = userBasedRbm.getHiddenActivitiesFromVisibleData(data);
//        userNegPhaseVisible = userBasedRbm.getVisibleActivitiesFromHiddenData(userHiddenActivities);
//        // null values missing in original data
//        RbmBasedCF.nullMissingValues(data, userNegPhaseVisible);
//        
//        itemHiddenActivities = itemBasedRbm.getHiddenActivitiesFromVisibleData(transposedData);
//        itemNegPhaseVisible = itemBasedRbm.getVisibleActivitiesFromHiddenData(itemHiddenActivities);
//        // null values missing in original data
//        RbmBasedCF.nullMissingValues(transposedData, itemNegPhaseVisible);
//        
//        for (int i=0; i<negPhaseVisible.length; i++) {
//        	for (int j=0; j<negPhaseVisible[i].length; j++) {
//        		negPhaseVisible[i][j] = (userNegPhaseVisible[i][j] + itemNegPhaseVisible[j][i]) / 2.0;
//        	}
//        }
//        
//        return Matrix.getMeanSquaredError(data, negPhaseVisible);
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
